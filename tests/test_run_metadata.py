import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from p2m.config import ConfigError, load_runtime_context
from p2m.runner import run_pipeline
from p2m.stages import STAGES


class RuntimeContextTest(unittest.TestCase):
    def test_load_runtime_context_supports_target_strings(self) -> None:
        context = load_runtime_context(
            {
                "suite_id": "suite-a",
                "run_id": "run-a",
                "risk": "harmful_medical_advice",
                "pipeline": {
                    "seeds": {"prompt": {"model": {"name": "azure/gpt-5.4"}}},
                    "rollout": {"target": {"model": {"name": "azure/gpt-5.4"}}},
                    "judge": {"judge": {"model": {"name": "azure/gpt-5.4"}}},
                },
            },
            Path("examples/pipes/health_assistant.yaml"),
            stage_modules=STAGES,
        )

        self.assertEqual(context["target"].model, "azure/gpt-5.4")
        self.assertEqual(context["run_id"], "run-a")

    def test_load_runtime_context_rejects_missing_rollout_target(self) -> None:
        with self.assertRaisesRegex(ConfigError, "pipeline.rollout.target is required"):
            load_runtime_context(
                {
                    "suite_id": "suite-a",
                    "risk": "harmful_medical_advice",
                    "pipeline": {
                        "rollout": {"seed_path": "seeds.jsonl"},
                        "judge": {"judge": {"model": {"name": "azure/gpt-5.4"}}},
                    },
                },
                Path("examples/pipes/health_assistant.yaml"),
                stage_modules=STAGES,
            )

    def test_load_runtime_context_allows_disabled_stage_family(self) -> None:
        context = load_runtime_context(
            {
                "suite_id": "suite-a",
                "risk": "harmful_medical_advice",
                "pipeline": {
                    "policy": {"enabled": False, "model": {"name": "azure/gpt-5.4"}},
                    "systematization": {"concept": "harmful medical advice"},
                },
            },
            Path("examples/pipes/health_assistant.yaml"),
            stage_modules=STAGES,
        )

        self.assertEqual(
            [stage_name for stage_name, _ in context["stages"]],
            ["policy", "systematization"],
        )


class RunnerManifestTest(unittest.TestCase):
    def test_run_pipeline_writes_minimal_manifest(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cfg_path = root / "config.yaml"
            suite_root = root / "results" / "suite-a"
            suite_root.mkdir(parents=True)
            transcripts_path = suite_root / "transcripts.jsonl"
            transcripts_path.write_text(
                '{"kind":"prompt","seed_id":"seed-1"}\n',
                encoding="utf-8",
            )
            (suite_root / "policy.json").write_text("{}", encoding="utf-8")
            cfg_path.write_text(
                "\n".join(
                    [
                        "suite_id: suite-a",
                        "run_id: run-a",
                        f"results_dir: {root / 'results'}",
                        "risk: harmful_medical_advice",
                        "pipeline:",
                        "  judge:",
                        "    judge:",
                        "      model:",
                        "        name: azure/gpt-5.4",
                        f"    transcripts_path: {transcripts_path}",
                    ]
                ),
                encoding="utf-8",
            )

            async def fake_run_judge(**_: object) -> dict[str, str]:
                run_root = root / "results" / "suite-a" / "run-a"
                run_root.mkdir(parents=True, exist_ok=True)
                scores = run_root / "scores.jsonl"
                metrics = run_root / "metrics.json"
                scores.write_text("", encoding="utf-8")
                metrics.write_text("{}", encoding="utf-8")
                return {
                    "scores_path": str(scores),
                    "metrics_path": str(metrics),
                }

            with patch("p2m.stages.judge.run_judge", new=fake_run_judge):
                rc = run_pipeline(config=str(cfg_path))

            self.assertEqual(rc, 0)
            manifest = json.loads(
                (root / "results" / "suite-a" / "run-a" / "manifest.json").read_text(encoding="utf-8")
            )
            suite_meta = json.loads(
                (root / "results" / "suite-a" / "suite.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                sorted(manifest.keys()),
                ["ended_at", "stages", "started_at", "status"],
            )
            self.assertEqual(manifest["stages"]["judge"], "completed")
            self.assertEqual(manifest["status"], "completed")
            self.assertIsInstance(manifest["started_at"], str)
            self.assertIn("T", manifest["started_at"])
            self.assertIsInstance(manifest["ended_at"], str)
            self.assertIn("T", manifest["ended_at"])
            self.assertEqual(sorted(suite_meta.keys()), ["created_at"])
            self.assertIn("T", suite_meta["created_at"])
            saved_config = root / "results" / "suite-a" / "run-a" / "config.yaml"
            self.assertTrue(saved_config.exists())
            self.assertEqual(saved_config.read_text(encoding="utf-8"), cfg_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
