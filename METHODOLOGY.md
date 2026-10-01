# DREAM-001: timing and evaluation

## Annotations and timing

Each JSONL record has `prompt`, `answer`, and exactly one of:

- `first_false_char`: zero-based **Unicode character** offset, or `null`.
- `false_span`: exact unique substring, `[start, end)` character offsets, or `null`.

`null` explicitly means no observed false claim. Missing labels and ambiguous
substrings are errors. Optional `id` must be unique. Optional `censored: true`
marks an incomplete/truncated answer; without an observed onset, its last N−1
positions have unknown targets and are excluded. Complete correct answers remain
negative through termination: the target concerns false claims in that answer.
Annotate the earliest false claim, not a later convenient span.

For zero-based answer tokens, the first token overlapping the onset is `t_H`.
Whitespace-only gaps map to the next token. Prompt and answer are tokenized
separately, matching a generated continuation; no answer-end marker is appended.
The prompt uses the model's single-user chat template, with one assistant prefix
and no duplicated BOS. To replay an existing experiment faithfully, match its
prompt template/model/precision; extend `prefix_ids` for system or multi-turn messages.

Row `i` is recorded **before token i**, after observing tokens through `t=i−1`:

```text
prompt + answer[:i] -> next-token distribution P_i and final state h_i
label = 1 iff t < t_H <= t + N   (equivalently i <= t_H < i + N)
relative_to_onset = t - t_H     (−1 is immediately before the false token)
```

There is also a prompt-only row (`t=−1`). The LM never sees token i or later
tokens when computing row i. A replay of externally supplied answers measures
the chosen model's conditional trajectory, not the original generator's states.

The classifier uses only these seven features, all from unmodified full-vocabulary
logits at temperature 1, with probability/transition calculations in FP32:

| Feature | Definition |
| --- | --- |
| `entropy` | −Σ P_i log P_i, in nats |
| `entropy_delta` | H_i − H_(i−1) |
| `logit_margin` | Largest minus second-largest raw logit |
| `kl_previous` | KL(P_i ∥ P_(i−1)), in nats |
| `hidden_step_norm` | ‖v_i‖₂, where v_i = h_i − h_(i−1) |
| `direction_consistency` | cos(v_i, v_(i−1)) |
| `curvature` | ‖v_i − v_(i−1)‖₂ |

`h_i` is the returned final-layer hidden state at the last observed position,
including final normalization as returned by the LM. Missing initial transitions
and zero-norm cosines are zero. No future smoothing, token identity, answer length,
offset, or onset metadata enters the classifier. This is N-token cumulative risk,
often called a horizon hazard; it is not an instantaneous survival hazard or a
guaranteed calibrated probability.

## Evaluation and files

All rows for an answer stay together; exact duplicate answer strings also share
a group. Unique groups are stratified by whether they contain a positive target.
The standardizer and unweighted logistic regression fit only training groups in
each fold. At least two positive and two negative-only groups are required;
fold count reduces to the smaller group-class count if necessary. Post-onset rows
are excluded from training and metrics. Censored unknown tail rows are excluded
too. Both still receive held-out predictions when their group participates.

`metrics.json` reports pooled **out-of-fold** ROC-AUC, average precision, positive
token base rate, onset recall, false alarms per 1,000 evaluated pre-token states,
and mean/median warning lead in tokens. An alarm is the start of a contiguous
above-threshold episode, reset at each answer. A false alarm has no onset within
N tokens; a detected onset needs an alarm start within its preceding N-token
window. Lead is `t_H−t`, so it ranges from 1 to N. Lead statistics condition on
detected onsets; missed onsets remain visible in recall. A warning that starts
too early and stays high is a false alarm, without a second onset warning.
No detections means `null` lead time. The threshold is fixed by the command,
never selected using held-out predictions.

- `features.csv`: all pre-token features, offsets, `t_H`, alignment, and targets
  (blank means excluded).
- `predictions.csv`: same rows plus `risk_oof`, `fold`, and `alarm_start_oof`.
- `metrics.json`: metrics and explicit train/test group membership for auditing.
- `extraction.json`: model/revision, data hash, runtime, and memory provenance.
- `hazard.joblib`: final all-eligible-data refit (`pipeline`, `features`, `horizon`);
  evaluation always uses the separate OOF predictions.
- `report.html`: self-contained interactive dashboard built from the OOF results.

This is a new exploratory hypothesis, not a reversal of DREAM-001's original
null result. Use fresh, verified onset labels and enough substantive correct
controls; match topic, position, difficulty, length, and refusal behavior. Tokens
and overlapping windows are correlated, so pooled token metrics do not provide
independent-sample significance. Answer grouping prevents within-answer leakage;
question/topic grouping and answer-level confidence intervals are further steps
for a confirmatory study.

API references: [HF chat templates](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating),
[model outputs](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/output),
[4-bit loading](https://huggingface.co/docs/transformers/v4.57.1/en/quantization/bitsandbytes),
[Llama model card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).
