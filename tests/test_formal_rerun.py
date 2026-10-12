import json
import tempfile
import unittest
from pathlib import Path
from robust_likelihood.formal_rerun import make_prompt_pack
from robust_likelihood.runner import run_comparison
from robust_likelihood.storage import repo_root


def specification():
    return {"unit": "complete_utterance", "reference_distribution": "TEST ONLY: a finite reference generator for complete utterances.", "epsilon_nats": 0.25}


class FormalRerunTests(unittest.TestCase):
    def test_definition_and_output_meaning(self):
        pack = make_prompt_pack(repo_root(), specification())
        self.assertIn("E[V~p(.|H)]", pack.pairwise_question)
        self.assertIn("not the emission probability", pack.pairwise_question)
        self.assertIn("<<UTTERANCE>>", pack.pairwise_question)

    def test_cache_identity_includes_reference_and_threshold(self):
        s = specification()
        a = make_prompt_pack(repo_root(), s).version
        s["epsilon_nats"] = 0.5
        b = make_prompt_pack(repo_root(), s).version
        s["reference_distribution"] += " Different sampling rule."
        c = make_prompt_pack(repo_root(), s).version
        self.assertEqual(len({a, b, c}), 3)

    def test_invalid_thresholds_and_missing_reference(self):
        for epsilon in [None, 0, -1, float("nan"), float("inf"), True]:
            with self.subTest(epsilon=epsilon):
                s = specification()
                s["epsilon_nats"] = epsilon
                with self.assertRaises(ValueError):
                    make_prompt_pack(repo_root(), s)
        s = specification()
        s["reference_distribution"] = None
        with self.assertRaises(ValueError):
            make_prompt_pack(repo_root(), s)

    def test_same_alarm_plan_pairwise_only_without_api(self):
        root = repo_root()
        pack = make_prompt_pack(root, specification())
        summary = run_comparison(root, providers=("jev", "gpt", "claude"), dataset="synthetic_v2", dry_run=True, prompt_pack=pack)
        baseline = run_comparison(root, providers=("jev", "gpt", "claude"), dataset="synthetic_v2", dry_run=True, prompt_version="typical")
        self.assertEqual(summary["api_calls"], 0)
        self.assertEqual(summary["planned_requests"]["classification"], 0)
        self.assertEqual(summary["planned_requests"]["pairwise"], baseline["planned_requests"]["pairwise"])
        self.assertEqual(summary["prompt_version"], pack.version)
        self.assertIn("not P(utterance|intention)", summary["wording_note"])

    def test_zero_call_run_does_not_produce_intent_accuracy(self):
        root = repo_root()
        pack = make_prompt_pack(root, specification())
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_comparison(root, providers=("jev",), dataset="synthetic_v2", prompt_pack=pack, clients={}, max_requests=0, results_root=Path(tmp))
            metrics = json.loads((Path(summary["run_dir"]) / "metrics.json").read_text())
            self.assertEqual(metrics["evaluation_status"], "raw_scores_only")
            self.assertNotIn("accuracy", metrics)
            self.assertEqual(summary["http_requests"], 0)


if __name__ == "__main__":
    unittest.main()
