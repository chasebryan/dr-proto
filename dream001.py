#!/usr/bin/env python3
"""DREAM-001: causal, pre-token state features -> N-token first-onset risk.

Row i is observed BEFORE answer token i: t=i-1, including t=-1 (prompt only).
Only the seven FEATURES enter the classifier; onset/offset metadata are labels.
"""
import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["entropy", "entropy_delta", "logit_margin", "kl_previous",
            "hidden_step_norm", "direction_consistency", "curvature"]


def onset_char(record):
    """Require explicit annotations; never silently treat a missing label as true."""
    answer = record["answer"]
    keys = [k for k in ("first_false_char", "false_span") if k in record]
    if len(keys) != 1:
        raise ValueError("Provide exactly one of first_false_char or false_span (null = no false claim).")
    value = record[keys[0]]
    if value is None:
        return None
    if keys[0] == "false_span":
        if isinstance(value, str) and value:
            start = answer.find(value)
            if start < 0 or answer.find(value, start + 1) >= 0:
                raise ValueError("false_span must occur exactly once; use [start, end] if ambiguous.")
            value = start
        elif (isinstance(value, list) and len(value) == 2
              and all(type(x) is int for x in value)
              and 0 <= value[0] < value[1] <= len(answer)):
            value = value[0]
        else:
            raise ValueError("false_span must be a nonempty exact string, [start, end], or null.")
    if type(value) is not int or not 0 <= value < len(answer):
        raise ValueError("first_false_char must be a zero-based Unicode character offset inside answer.")
    return value


def load_records(path):
    records, seen = [], set()
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not all(isinstance(record.get(k), str) and record[k] for k in ("prompt", "answer")):
                raise ValueError("prompt and answer must be nonempty strings.")
            record["id"] = str(record.get("id", line_no))
            if record["id"] in seen:
                raise ValueError("Duplicate record id.")
            if type(record.get("censored", False)) is not bool:
                raise ValueError("censored must be a boolean.")
            record["onset_char"] = onset_char(record)
            seen.add(record["id"])
            records.append(record)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not records:
        raise ValueError("Empty annotation file.")
    return records


def onset_token(offsets, char):
    if char is None:
        return None
    # First overlapping token. Whitespace gaps map to the next nonempty token;
    # byte-fallback tokens may share offsets, so choose the first of them.
    for i, (start, end) in enumerate(offsets):
        if end > start and end > char:
            return i
    raise ValueError("Onset has no corresponding token (e.g. trailing whitespace).")


def target_at(i, first_false, length, horizon, censored=False):
    if first_false is not None:
        return int(i <= first_false < i + horizon) if i <= first_false else None
    # A truncated answer has unknown tail labels, not negative labels.
    return None if censored and i + horizon > length else 0


class StateFeatures:
    """Keep only the preceding distribution, state, and step; all math is FP32."""
    def __init__(self):
        self.logp = self.hidden = self.step = self.entropy = None

    def update(self, logits, hidden):
        import torch
        logp = torch.log_softmax(logits.float(), dim=-1)
        hidden = hidden.float().clone()
        if not torch.isfinite(logp).all() or not torch.isfinite(hidden).all():
            raise ValueError("Nonfinite model states/logits.")
        p = logp.exp()
        entropy = -(p * logp).sum().item()
        top = logits.float().topk(2).values
        values = [entropy, 0., (top[0] - top[1]).item(), 0., 0., 0., 0.]
        step = None
        if self.logp is not None:
            step = hidden - self.hidden
            values[1] = entropy - self.entropy
            values[3] = max(0., (p * (logp - self.logp)).sum().item())
            values[4] = step.norm().item()
            if self.step is not None:
                denom = step.norm() * self.step.norm()
                values[5] = (torch.dot(step, self.step) / denom).clamp(-1, 1).item() if denom > 0 else 0.
                values[6] = (step - self.step).norm().item()
        self.logp, self.hidden, self.step, self.entropy = logp, hidden, step, entropy
        return dict(zip(FEATURES, values))


def prefix_ids(tokenizer, prompt, plain=False):
    if plain:
        ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    else:
        if not tokenizer.chat_template:
            raise ValueError("Tokenizer has no chat template; use --plain for a base LM.")
        # tokenize=True includes the template's BOS exactly once.
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                             tokenize=True, add_generation_prompt=True)
    if not ids:
        raise ValueError("Prompt produced no tokens.")
    return ids


def trace_record(model, tokenizer, record, device, horizon=8, max_tokens=1024,
                 prefill_chunk=64, plain=False):
    """Teacher-force one token at a time; never feed answer[i] before row i."""
    import torch
    prefix = prefix_ids(tokenizer, record["prompt"], plain)
    answer = tokenizer(record["answer"], add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = answer["input_ids"], answer["offset_mapping"]
    if not ids:
        raise ValueError(f"{record['id']}: answer produced no tokens.")
    if len(prefix) + len(ids) > max_tokens:
        raise ValueError(f"{record['id']}: {len(prefix) + len(ids)} tokens exceed --max-tokens {max_tokens}; no silent truncation.")
    first_false = onset_token(offsets, record["onset_char"])
    group = hashlib.sha256(record["answer"].encode("utf-8")).hexdigest()
    cache, consumed = None, 0
    # Supported models avoid projecting every prefill position over the vocabulary.
    parameters = inspect.signature(model.forward).parameters
    keep_key = next((k for k in ("logits_to_keep", "num_logits_to_keep") if k in parameters), None)

    def forward(tokens, states=True):
        nonlocal cache, consumed
        consumed += len(tokens)
        kwargs = {keep_key: 1} if keep_key else {}
        output = model(input_ids=torch.tensor([tokens], device=device),
                       attention_mask=torch.ones((1, consumed), dtype=torch.long, device=device),
                       past_key_values=cache, use_cache=True, output_hidden_states=states,
                       return_dict=True, **kwargs)
        cache = output.past_key_values
        return output

    rows, state = [], StateFeatures()
    with torch.inference_mode():
        for start in range(0, len(prefix), prefill_chunk):
            chunk = prefix[start:start + prefill_chunk]
            output = forward(chunk, states=start + len(chunk) == len(prefix))
            if start + len(chunk) < len(prefix):
                del output
        for i, token_id in enumerate(ids):
            features = state.update(output.logits[0, -1], output.hidden_states[-1][0, -1])
            # Free prefill layer states/logits before the next forward pass.
            del output
            char_start, char_end = offsets[i]
            rows.append({"record_id": record["id"], "answer_group": group,
                         "t": i - 1, "next_token_index": i, "token_id": token_id,
                         "token_text": record["answer"][char_start:char_end],
                         "char_start": char_start, "char_end": char_end,
                         "answer_tokens": len(ids), "first_false_char": record["onset_char"],
                         "first_false_token": first_false,
                         "relative_to_onset": None if first_false is None else i - 1 - first_false,
                         "at_risk": int(first_false is None or i <= first_false),
                         "censored": int(record.get("censored", False)), "horizon": horizon,
                         "target": target_at(i, first_false, len(ids), horizon, record.get("censored", False)),
                         **features})
            if i + 1 < len(ids):
                output = forward([token_id])
    return rows


def load_lm(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if args.four_bit and device != "cuda":
        raise ValueError("--4bit requires a CUDA GPU in this prototype.")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for character offsets.")
    kwargs = dict(revision=args.revision, dtype=dtype, low_cpu_mem_usage=True,
                  attn_implementation="sdpa", trust_remote_code=False)
    if args.four_bit:
        kwargs.update(device_map={"": device}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype))
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if not args.four_bit:
        model.to(device)
    model.eval()
    return model, tokenizer, device, str(dtype)


def extract(args):
    import torch
    import transformers
    records = load_records(args.annotations)
    model, tokenizer, device, dtype = load_lm(args)
    if args.max_tokens > getattr(model.config, "max_position_embeddings", args.max_tokens):
        raise ValueError("--max-tokens exceeds model context limit.")
    rows = []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for j, record in enumerate(records, 1):
        rows.extend(trace_record(model, tokenizer, record, device, args.horizon,
                                 args.max_tokens, args.prefill_chunk, args.plain))
        print(f"Extracted {j}/{len(records)} answers", file=sys.stderr, flush=True)
    frame = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "features.csv", index=False)
    provenance = {"model": args.model, "revision_requested": args.revision,
                  "revision_resolved": getattr(model.config, "_commit_hash", None),
                  "annotations_sha256": hashlib.sha256(Path(args.annotations).read_bytes()).hexdigest(),
                  "device": device, "dtype": dtype, "four_bit": args.four_bit,
                  "plain_prompt": args.plain, "horizon": args.horizon,
                  "max_tokens": args.max_tokens, "prefill_chunk": args.prefill_chunk,
                  "torch": torch.__version__, "transformers": transformers.__version__,
                  "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None}
    (out / "extraction.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return frame


def grouped_folds(frame, folds, seed):
    """Stratify unique answer groups, then expand to rows: never split an answer."""
    from sklearn.model_selection import StratifiedKFold
    eligible = frame[frame.target.notna()]
    groups = eligible.groupby("answer_group").target.max()
    counts = groups.value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError("Need at least two distinct positive answer groups and two negative-only groups.")
    count = min(folds, int(counts.min()))
    splitter = StratifiedKFold(n_splits=count, shuffle=True, random_state=seed)
    for train, test in splitter.split(groups.index, groups.to_numpy()):
        train_mask = frame.answer_group.isin(groups.index[train]) & frame.target.notna()
        test_mask = frame.answer_group.isin(groups.index[test])
        yield np.flatnonzero(train_mask), np.flatnonzero(test_mask)


def warning_metrics(frame, probabilities, threshold):
    """One alarm per contiguous above-threshold episode, reset for each answer."""
    alarms = np.zeros(len(frame), dtype=int)
    leads, false_alarms, onsets = [], 0, 0
    for _, answer in frame.groupby("record_id", sort=False):
        onsets += int(answer.first_false_token.notna().any())
        valid = answer[answer.target.notna()].sort_values("next_token_index")
        high = probabilities[valid.index] >= threshold
        starts = high & ~np.r_[False, high[:-1]]
        alarm_rows = valid.loc[starts]
        alarms[alarm_rows.index] = 1
        false_alarms += int((alarm_rows.target == 0).sum())
        true = alarm_rows[alarm_rows.target == 1]
        if not true.empty:
            first = true.iloc[0]
            leads.append(int(first.first_false_token - first.t))
    evaluated = int(frame.target.notna().sum())
    return {"alarm_threshold": threshold, "false_alarms": false_alarms,
            "false_alarms_per_1000_tokens": 1000 * false_alarms / evaluated,
            "false_alarm_denominator_tokens": evaluated,
            "onsets_total": onsets, "onsets_detected": len(leads),
            "onset_recall": len(leads) / onsets if onsets else None,
            "warning_lead_tokens_mean": float(np.mean(leads)) if leads else None,
            "warning_lead_tokens_median": float(np.median(leads)) if leads else None}, alarms


def fit(frame, args):
    import joblib
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    frame = frame.reset_index(drop=True).copy()
    required = FEATURES + ["record_id", "answer_group", "target", "horizon",
                           "first_false_token", "next_token_index", "t"]
    if any(k not in frame for k in required):
        raise ValueError("Input is missing required feature/metadata columns.")
    if frame.horizon.nunique() != 1 or not frame.target.dropna().isin([0, 1]).all():
        raise ValueError("CSV must have one horizon and binary/blank targets.")
    X = frame[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Features must be finite.")
    probability, fold_id = np.full(len(frame), np.nan), np.full(len(frame), -1)
    def estimator():
        # No class balancing: preserve the empirical token-risk prior.
        return make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000, random_state=args.seed))
    audit = []
    for fold, (train, test) in enumerate(grouped_folds(frame, args.folds, args.seed)):
        classifier = estimator()
        classifier.fit(X[train], frame.target.iloc[train].astype(int))
        probability[test] = classifier.predict_proba(X[test])[:, 1]
        fold_id[test] = fold
        audit.append({"fold": fold, "train_answer_groups": sorted(frame.answer_group.iloc[train].unique()),
                      "test_answer_groups": sorted(frame.answer_group.iloc[test].unique())})
    valid = frame.target.notna().to_numpy()
    if not np.isfinite(probability[valid]).all():
        raise ValueError("Some eligible tokens did not receive out-of-fold predictions.")
    y = frame.target[valid].astype(int).to_numpy()
    metrics, alarms = warning_metrics(frame, probability, args.threshold)
    metrics.update({"roc_auc_oof": float(roc_auc_score(y, probability[valid])),
                    "average_precision_oof": float(average_precision_score(y, probability[valid])),
                    "positive_token_base_rate": float(y.mean()), "evaluated_tokens": len(y),
                    "excluded_tokens": int((~valid).sum()), "answer_records": frame.record_id.nunique(),
                    "folds_requested": args.folds, "folds_used": len(audit), "seed": args.seed,
                    "horizon": int(frame.horizon.iloc[0]), "features": FEATURES,
                    "sklearn": sklearn.__version__, "fold_audit": audit})
    frame["risk_oof"], frame["fold"], frame["alarm_start_oof"] = probability, fold_id, alarms
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    final = estimator().fit(X[valid], y)
    joblib.dump({"pipeline": final, "features": FEATURES, "horizon": metrics["horizon"]}, out / "hazard.joblib")
    from report import write_report
    write_report(frame, metrics, out)
    summary = {k: v for k, v in metrics.items() if k not in ("fold_audit", "features")}
    print(json.dumps(summary, indent=2))
    return frame, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "extract", "fit"):
        p = sub.add_parser(command)
        p.add_argument("--out", required=True, help="Output directory")
        if command != "fit":
            p.add_argument("--annotations", required=True)
            p.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
            p.add_argument("--revision", default="main", help="Use a commit SHA for reproducibility")
            p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
            p.add_argument("--4bit", dest="four_bit", action="store_true")
            p.add_argument("--plain", action="store_true", help="Raw prompt for a base model")
            p.add_argument("--horizon", type=int, default=8)
            p.add_argument("--max-tokens", type=int, default=1024, help="Prompt + answer; fail rather than truncate")
            p.add_argument("--prefill-chunk", type=int, default=64)
        else:
            p.add_argument("--features", required=True)
        if command != "extract":
            p.add_argument("--folds", type=int, default=5)
            p.add_argument("--threshold", type=float, default=.5, help="Fixed in advance; never tuned on OOF scores")
            p.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command != "fit" and min(args.horizon, args.max_tokens, args.prefill_chunk) < 1:
        parser.error("horizon, max-tokens, and prefill-chunk must be positive.")
    if args.command != "extract" and (args.folds < 2 or not 0 < args.threshold < 1):
        parser.error("folds must be >= 2 and threshold strictly between 0 and 1.")
    try:
        frame = pd.read_csv(args.features, dtype={"record_id": str, "answer_group": str}) if args.command == "fit" else extract(args)
        if args.command != "extract":
            fit(frame, args)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"DREAM-001: {exc}\n")


if __name__ == "__main__":
    main()
