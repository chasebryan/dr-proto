"""Offline tests using a tiny randomly initialized Llama, never real-model scores."""
import argparse
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dream001 as d


def tiny_lm():
    # Character tokens give known offsets, including Unicode and whitespace.
    chars = sorted(set(ROOT.joinpath("example_annotations.jsonl").read_text()) | set("é"))
    vocab = {c: i for i, c in enumerate(["<unk>", "<bos>", "<eos>"] + chars)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                        bos_token="<bos>", eos_token="<eos>")
    tokenizer.chat_template = "{{ bos_token }}User: {{ messages[0]['content'] }}\nAssistant: "
    torch.manual_seed(7)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=32,
                            intermediate_size=64, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=1024, bos_token_id=1, eos_token_id=2))
    return model.eval(), tokenizer


class DreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model, cls.tokenizer = tiny_lm()

    def record(self, answer="Paris. Berlin.", span="Berlin"):
        return {"id": "a", "prompt": "France?", "answer": answer,
                "onset_char": None if span is None else answer.index(span)}

    def test_annotation_validation_unicode_and_ambiguity(self):
        self.assertEqual(d.onset_char({"answer": "é Paris", "false_span": "Paris"}), 2)
        self.assertEqual(d.onset_char({"answer": "é Paris", "false_span": [2, 7]}), 2)
        for r in [{"answer": "Paris"}, {"answer": "aa", "false_span": "a"},
                  {"answer": "a", "first_false_char": True},
                  {"answer": "a", "first_false_char": 1}]:
            with self.assertRaises(ValueError):
                d.onset_char(r)
        self.assertEqual(d.onset_token([(0, 1), (2, 4), (2, 4)], 2), 1)
        self.assertEqual(d.onset_token([(0, 1), (2, 4)], 1), 1)

    def test_horizon_boundaries_and_censoring(self):
        self.assertEqual([d.target_at(i, 4, 7, 2) for i in range(7)],
                         [0, 0, 0, 1, 1, None, None])
        self.assertEqual(d.target_at(0, 0, 7, 2), 1)
        self.assertEqual([d.target_at(i, None, 4, 3, True) for i in range(4)],
                         [0, 0, None, None])
        self.assertEqual(d.target_at(3, None, 4, 3, False), 0)

    def test_transition_formulas(self):
        state = d.StateFeatures()
        logits = torch.tensor([0., 0.])
        a = state.update(logits, torch.tensor([0., 0.]))
        b = state.update(logits, torch.tensor([1., 0.]))
        c = state.update(logits, torch.tensor([1., 1.]))
        self.assertAlmostEqual(a["entropy"], np.log(2), places=6)
        self.assertEqual(b["kl_previous"], 0)
        self.assertEqual(b["hidden_step_norm"], 1)
        self.assertEqual(c["direction_consistency"], 0)
        self.assertAlmostEqual(c["curvature"], np.sqrt(2), places=6)
        changed = state.update(torch.tensor([2., 0.]), torch.tensor([1., 2.]))
        p = torch.softmax(torch.tensor([2., 0.]), -1)
        expected_kl = (p * (p.log() - np.log(.5))).sum().item()
        self.assertAlmostEqual(changed["kl_previous"], expected_kl, places=6)
        self.assertEqual(changed["logit_margin"], 2)
        self.assertEqual(changed["direction_consistency"], 1)
        self.assertEqual(changed["curvature"], 0)

    def test_streaming_matches_full_causal_forward_and_alignment(self):
        record = self.record()
        rows = d.trace_record(self.model, self.tokenizer, record, "cpu", horizon=3, prefill_chunk=3)
        prefix = d.prefix_ids(self.tokenizer, record["prompt"])
        ids = self.tokenizer(record["answer"], add_special_tokens=False)["input_ids"]
        self.assertEqual(prefix.count(self.tokenizer.bos_token_id), 1)
        with torch.inference_mode():
            full = self.model(torch.tensor([prefix + ids]), output_hidden_states=True)
            state = d.StateFeatures()
            for i, row in enumerate(rows):
                # Predictor of token i is at prefix_len+i-1, not prefix_len+i.
                pos = len(prefix) + i - 1
                expected = state.update(full.logits[0, pos], full.hidden_states[-1][0, pos])
                np.testing.assert_allclose([row[k] for k in d.FEATURES],
                                           [expected[k] for k in d.FEATURES], atol=2e-5)
        h = record["onset_char"]  # This fixture uses one token per character.
        self.assertEqual(rows[h]["first_false_token"], h)
        self.assertEqual(rows[h]["t"], h - 1)
        self.assertEqual(rows[h]["relative_to_onset"], -1)
        self.assertEqual(rows[h]["target"], 1)
        self.assertIsNone(rows[h + 1]["target"])

    def test_future_suffix_cannot_change_pre_token_features(self):
        left = d.trace_record(self.model, self.tokenizer, self.record("Paris. Berlin."), "cpu")
        right = d.trace_record(self.model, self.tokenizer, self.record("Paris. France.", "France"), "cpu")
        boundary = len("Paris. ")
        # Includes the row predicting the FIRST different token itself.
        np.testing.assert_array_equal([[r[k] for k in d.FEATURES] for r in left[:boundary + 1]],
                                      [[r[k] for k in d.FEATURES] for r in right[:boundary + 1]])
        with self.assertRaisesRegex(ValueError, "no silent truncation"):
            d.trace_record(self.model, self.tokenizer, self.record(), "cpu", max_tokens=2)

    def test_alarm_episode_metrics_and_misses(self):
        rows = []
        for rid, onset, length in [("bad", 4, 7), ("good", None, 5)]:
            rows.extend({"record_id": rid, "next_token_index": i, "t": i - 1,
                         "first_false_token": onset, "target": d.target_at(i, onset, length, 2)}
                        for i in range(length))
        frame = pd.DataFrame(rows)
        p = np.array([.7, .8, .1, .9, .8, .9, .9, .1, .7, .8, .1, .6])
        metrics, alarms = d.warning_metrics(frame, p, .5)
        self.assertEqual(metrics["false_alarms"], 3)
        self.assertEqual(metrics["false_alarms_per_1000_tokens"], 300)
        self.assertEqual(metrics["warning_lead_tokens_mean"], 2)
        self.assertEqual(metrics["onset_recall"], 1)
        self.assertEqual(np.flatnonzero(alarms).tolist(), [0, 3, 8, 11])
        missed, _ = d.warning_metrics(frame, np.ones(len(frame)), .5)
        self.assertEqual(missed["onset_recall"], 0)
        self.assertIsNone(missed["warning_lead_tokens_mean"])

    def test_offline_report_embeds_data_without_script_injection(self):
        from report import write_report
        frame = pd.DataFrame([{"token_text": "</script><script>bad()</script>", "target": np.nan}])
        with tempfile.TemporaryDirectory() as directory:
            path = write_report(frame, {"example": 1}, directory, {"demo": True})
            html = path.read_text()
            embedded = html.split('id="dream-data">')[1].split('</script>')[0]
            data = json.loads(embedded)
            self.assertNotIn("</script>", embedded)
            self.assertEqual(data["rows"][0]["token_text"], frame.token_text.iloc[0])
            self.assertIsNone(data["rows"][0]["target"])

    def test_complete_offline_cli_and_duplicate_answer_grouping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.model.save_pretrained(root / "model")
            self.tokenizer.save_pretrained(root / "model")
            cmd = [sys.executable, str(ROOT / "dream001.py"), "run", "--annotations",
                   str(ROOT / "example_annotations.jsonl"), "--model", str(root / "model"),
                   "--device", "cpu", "--horizon", "8", "--folds", "3", "--out", str(root / "run")]
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            self.assertEqual(run.returncode, 0, run.stderr)
            metrics = json.loads((root / "run/metrics.json").read_text())
            self.assertEqual(metrics["answer_records"], 8)
            self.assertEqual(metrics["onsets_total"], 4)
            self.assertTrue(0 <= metrics["roc_auc_oof"] <= 1)
            self.assertTrue((root / "run/hazard.joblib").exists())
            self.assertTrue((root / "run/report.html").exists())
            frame = pd.read_csv(root / "run/predictions.csv")
            self.assertTrue(frame.risk_oof.notna().all())
            self.assertTrue((frame.groupby("record_id").fold.nunique() == 1).all())
            duplicate = frame[frame.record_id == "paris_false"].copy()
            duplicate["record_id"] = "duplicate"
            frame = pd.concat([frame, duplicate], ignore_index=True)
            args = argparse.Namespace(folds=5, seed=42, threshold=.5, out=root / "refit")
            with contextlib.redirect_stdout(io.StringIO()):
                result, metrics = d.fit(frame, args)
            self.assertEqual(metrics["folds_used"], 4)
            self.assertTrue((result.groupby("answer_group").fold.nunique() == 1).all())
            for fold in metrics["fold_audit"]:
                self.assertFalse(set(fold["train_answer_groups"]) & set(fold["test_answer_groups"]))


if __name__ == "__main__":
    unittest.main()
