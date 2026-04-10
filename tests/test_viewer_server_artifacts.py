import json
import os
import shutil
import subprocess
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
DATA_SRC = ROOT / "viewer" / "src" / "lib" / "server" / "data.ts"
METRICS_SRC = ROOT / "viewer" / "src" / "lib" / "server" / "metrics.ts"
DIMENSIONS_SRC = ROOT / "viewer" / "src" / "lib" / "server" / "dimensions.ts"
ARTIFACTS_SRC = ROOT / "viewer" / "src" / "lib" / "server" / "artifacts.ts"
CONFIG_SRC = ROOT / "viewer" / "src" / "lib" / "server" / "config.ts"
JUDGMENT_SRC = ROOT / "viewer" / "src" / "lib" / "judgment.ts"


class ViewerServerArtifactsTest(unittest.TestCase):
    def _copy_data_harness(self, harness_dir: Path) -> Path:
        data_path = harness_dir / "data.ts"
        metrics_path = harness_dir / "metrics.ts"
        dimensions_path = harness_dir / "dimensions.ts"
        artifacts_path = harness_dir / "artifacts.ts"
        config_path = harness_dir / "config.ts"
        judgment_path = harness_dir / "judgment.ts"

        data_source = (
            DATA_SRC.read_text(encoding="utf-8")
            .replace("./config.js", "./config.ts")
            .replace("./dimensions.js", "./dimensions.ts")
            .replace("./artifacts.js", "./artifacts.ts")
            .replace("./metrics.js", "./metrics.ts")
            .replace("$lib/judgment.js", "./judgment.ts")
        )
        metrics_source = (
            METRICS_SRC.read_text(encoding="utf-8")
            .replace("$lib/judgment.js", "./judgment.ts")
            .replace("./dimensions.js", "./dimensions.ts")
        )
        dimensions_source = (
            DIMENSIONS_SRC.read_text(encoding="utf-8")
            .replace("./config.js", "./config.ts")
            .replace("./artifacts.js", "./artifacts.ts")
        )
        artifacts_source = ARTIFACTS_SRC.read_text(encoding="utf-8").replace(
            "./config.js", "./config.ts"
        )

        data_path.write_text(data_source, encoding="utf-8")
        metrics_path.write_text(metrics_source, encoding="utf-8")
        dimensions_path.write_text(dimensions_source, encoding="utf-8")
        artifacts_path.write_text(artifacts_source, encoding="utf-8")
        shutil.copyfile(CONFIG_SRC, config_path)
        shutil.copyfile(JUDGMENT_SRC, judgment_path)
        return data_path

    def _run_node(
        self, *, harness_dir: Path, script: str, env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["node", "--experimental-strip-types", "--input-type=module"],
            input=script,
            text=True,
            capture_output=True,
            cwd=harness_dir,
            env=env,
            check=False,
        )

    def test_load_judged_samples_surfaces_invalid_scores_jsonl(self) -> None:
        with TemporaryDirectory(dir=ROOT / "viewer") as tmp_dir:
            tmp_root = Path(tmp_dir)
            harness_dir = tmp_root / "harness"
            harness_dir.mkdir()
            data_path = self._copy_data_harness(harness_dir)

            artifacts_root = tmp_root / "artifacts" / "results"
            suite_dir = artifacts_root / "suite-a"
            run_dir = suite_dir / "run-a"
            run_dir.mkdir(parents=True, exist_ok=True)

            (suite_dir / "seeds.jsonl").write_text(
                json.dumps(
                    {
                        "kind": "prompt",
                        "seed_id": "seed-1",
                        "permissible": False,
                        "risk": "risk",
                        "sub_risk": "sub-risk",
                        "definition": "def",
                        "seed": {"description": "prompt"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "scores.jsonl").write_text("{bad jsonl\n", encoding="utf-8")

            env = os.environ.copy()
            env.update(
                {
                    "ARTIFACTS_ROOT": str(artifacts_root),
                    "MEASUREMENTS_ROOT": str(tmp_root),
                }
            )
            script = textwrap.dedent(
                f"""\
                try {{
                  const {{ loadJudgedSamples }} = await import({json.dumps(data_path.as_uri())});
                  loadJudgedSamples('suite-a', 'run-a');
                  console.log(JSON.stringify({{ ok: true }}));
                }} catch (error) {{
                  console.log(JSON.stringify({{
                    ok: false,
                    name: error.name,
                    message: error.message
                  }}));
                }}
                """
            )
            result = self._run_node(harness_dir=harness_dir, script=script, env=env)

            self.assertEqual(result.returncode, 0, msg=f"{result.stdout}\n{result.stderr}")
            payload = json.loads(result.stdout)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["name"], "ArtifactParseError")
            self.assertIn("scores.jsonl", payload["message"])

    def test_load_dimensions_surfaces_invalid_yaml(self) -> None:
        with TemporaryDirectory(dir=ROOT / "viewer") as tmp_dir:
            tmp_root = Path(tmp_dir)
            harness_dir = tmp_root / "harness"
            harness_dir.mkdir()
            self._copy_data_harness(harness_dir)

            dimensions_dir = tmp_root / "examples" / "eval-definitions"
            dimensions_dir.mkdir(parents=True, exist_ok=True)
            (dimensions_dir / "judge_dimensions.yaml").write_text("bad: [yaml", encoding="utf-8")

            env = os.environ.copy()
            env.update(
                {
                    "ARTIFACTS_ROOT": str(tmp_root / "artifacts" / "results"),
                    "MEASUREMENTS_ROOT": str(tmp_root),
                }
            )
            script = textwrap.dedent(
                """\
                try {
                  const { loadDimensions } = await import('./dimensions.ts');
                  loadDimensions();
                  console.log(JSON.stringify({ ok: true }));
                } catch (error) {
                  console.log(JSON.stringify({
                    ok: false,
                    name: error.name,
                    message: error.message
                  }));
                }
                """
            )
            result = self._run_node(harness_dir=harness_dir, script=script, env=env)

            self.assertEqual(result.returncode, 0, msg=f"{result.stdout}\n{result.stderr}")
            payload = json.loads(result.stdout)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["name"], "ArtifactParseError")
            self.assertIn("judge_dimensions.yaml", payload["message"])

    def test_load_run_page_data_reads_live_transcripts_during_rollout(self) -> None:
        with TemporaryDirectory(dir=ROOT / "viewer") as tmp_dir:
            tmp_root = Path(tmp_dir)
            harness_dir = tmp_root / "harness"
            harness_dir.mkdir()
            data_path = self._copy_data_harness(harness_dir)

            artifacts_root = tmp_root / "artifacts" / "results"
            suite_dir = artifacts_root / "suite-a"
            run_dir = suite_dir / "run-a"
            run_dir.mkdir(parents=True, exist_ok=True)

            (suite_dir / "suite.json").write_text(
                json.dumps({"created_at": "2026-04-02T00:00:00Z"}),
                encoding="utf-8",
            )
            (suite_dir / "seeds.jsonl").write_text(
                json.dumps(
                    {
                        "kind": "scenario",
                        "seed_id": "seed-1",
                        "permissible": False,
                        "risk": "risk",
                        "sub_risk": "sub-risk",
                        "definition": "def",
                        "seed": {
                            "title": "Scenario title",
                            "description": "Scenario description",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "manifest.json").write_text(
                json.dumps({"status": "running", "stages": {"rollout": "running"}}),
                encoding="utf-8",
            )
            valid_row = {
                "kind": "scenario",
                "seed_id": "seed-1",
                "risk": "risk",
                "sub_risk": "sub-risk",
                "permissible": False,
                "stop_reason": "max_turns",
                "target": "target-model",
                "auditor_model": "auditor-model",
                "events": [
                    {
                        "view": ["target"],
                        "actor": "auditor",
                        "edit": {
                            "type": "set_system_message",
                            "message": {"role": "system", "content": "System prompt"},
                        },
                    },
                    {
                        "view": ["target"],
                        "actor": "auditor",
                        "edit": {
                            "type": "add_message",
                            "message": {"role": "user", "content": "Need advice"},
                        },
                    },
                    {
                        "view": ["target"],
                        "actor": "target",
                        "edit": {
                            "type": "add_message",
                            "message": {"role": "assistant", "content": "Response"},
                        },
                    },
                ],
                "llm_calls": [],
            }
            (run_dir / "transcripts.jsonl").write_text(
                json.dumps(valid_row) + "\n" + '{"kind":"scenario"',
                encoding="utf-8",
            )

            env = os.environ.copy()
            env.update(
                {
                    "ARTIFACTS_ROOT": str(artifacts_root),
                    "MEASUREMENTS_ROOT": str(tmp_root),
                }
            )
            script = textwrap.dedent(
                f"""\
                const {{ loadRunPageData }} = await import({json.dumps(data_path.as_uri())});
                const payload = loadRunPageData('suite-a', 'run-a');
                console.log(JSON.stringify({{
                  previewCount: payload.rolloutPreviewRows.length,
                  previewSeed: payload.rolloutPreviewRows[0]?.seed_id ?? null,
                  previewTurns: payload.rolloutPreviewRows[0]?.turns_count ?? null,
                  previewTotal: payload.rolloutPreviewTotal,
                  auditScores: payload.auditScores.length,
                  transcriptCount: payload.transcriptMap['seed-1']?.length ?? 0
                }}));
                """
            )
            result = self._run_node(harness_dir=harness_dir, script=script, env=env)

            self.assertEqual(result.returncode, 0, msg=f"{result.stdout}\n{result.stderr}")
            payload = json.loads(result.stdout)
            self.assertEqual(payload["previewCount"], 1)
            self.assertEqual(payload["previewSeed"], "seed-1")
            self.assertEqual(payload["previewTurns"], 2)
            self.assertEqual(payload["previewTotal"], 1)
            self.assertEqual(payload["auditScores"], 0)
            self.assertEqual(payload["transcriptCount"], 3)

    def test_load_run_page_data_rejects_malformed_interior_live_transcript_line(self) -> None:
        with TemporaryDirectory(dir=ROOT / "viewer") as tmp_dir:
            tmp_root = Path(tmp_dir)
            harness_dir = tmp_root / "harness"
            harness_dir.mkdir()
            data_path = self._copy_data_harness(harness_dir)

            artifacts_root = tmp_root / "artifacts" / "results"
            suite_dir = artifacts_root / "suite-a"
            run_dir = suite_dir / "run-a"
            run_dir.mkdir(parents=True, exist_ok=True)

            (suite_dir / "suite.json").write_text(
                json.dumps({"created_at": "2026-04-02T00:00:00Z"}),
                encoding="utf-8",
            )
            (suite_dir / "seeds.jsonl").write_text(
                json.dumps(
                    {
                        "kind": "scenario",
                        "seed_id": "seed-1",
                        "permissible": False,
                        "risk": "risk",
                        "sub_risk": "sub-risk",
                        "definition": "def",
                        "seed": {"title": "Scenario title", "description": "Scenario description"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "manifest.json").write_text(
                json.dumps({"status": "running", "stages": {"rollout": "running"}}),
                encoding="utf-8",
            )
            valid_row = {
                "kind": "scenario",
                "seed_id": "seed-1",
                "risk": "risk",
                "sub_risk": "sub-risk",
                "permissible": False,
                "stop_reason": "max_turns",
                "events": [],
                "llm_calls": [],
            }
            (run_dir / "transcripts.jsonl").write_text(
                json.dumps(valid_row) + "\n{bad jsonl\n" + json.dumps(valid_row) + "\n",
                encoding="utf-8",
            )

            env = os.environ.copy()
            env.update(
                {
                    "ARTIFACTS_ROOT": str(artifacts_root),
                    "MEASUREMENTS_ROOT": str(tmp_root),
                }
            )
            script = textwrap.dedent(
                f"""\
                try {{
                  const {{ loadRunPageData }} = await import({json.dumps(data_path.as_uri())});
                  loadRunPageData('suite-a', 'run-a');
                  console.log(JSON.stringify({{ ok: true }}));
                }} catch (error) {{
                  console.log(JSON.stringify({{
                    ok: false,
                    name: error.name,
                    message: error.message
                  }}));
                }}
                """
            )
            result = self._run_node(harness_dir=harness_dir, script=script, env=env)

            self.assertEqual(result.returncode, 0, msg=f"{result.stdout}\n{result.stderr}")
            payload = json.loads(result.stdout)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["name"], "ArtifactParseError")
            self.assertIn("transcripts.jsonl", payload["message"])
            self.assertIn("line 2", payload["message"])

    def test_load_run_page_data_rejects_truncated_trailing_line_after_rollout(self) -> None:
        with TemporaryDirectory(dir=ROOT / "viewer") as tmp_dir:
            tmp_root = Path(tmp_dir)
            harness_dir = tmp_root / "harness"
            harness_dir.mkdir()
            data_path = self._copy_data_harness(harness_dir)

            artifacts_root = tmp_root / "artifacts" / "results"
            suite_dir = artifacts_root / "suite-a"
            run_dir = suite_dir / "run-a"
            run_dir.mkdir(parents=True, exist_ok=True)

            (suite_dir / "suite.json").write_text(
                json.dumps({"created_at": "2026-04-02T00:00:00Z"}),
                encoding="utf-8",
            )
            (suite_dir / "seeds.jsonl").write_text(
                json.dumps(
                    {
                        "kind": "scenario",
                        "seed_id": "seed-1",
                        "permissible": False,
                        "risk": "risk",
                        "sub_risk": "sub-risk",
                        "definition": "def",
                        "seed": {"title": "Scenario title", "description": "Scenario description"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {"status": "running", "stages": {"rollout": "completed", "judge": "running"}}
                ),
                encoding="utf-8",
            )
            valid_row = {
                "kind": "scenario",
                "seed_id": "seed-1",
                "risk": "risk",
                "sub_risk": "sub-risk",
                "permissible": False,
                "stop_reason": "max_turns",
                "events": [],
                "llm_calls": [],
            }
            (run_dir / "transcripts.jsonl").write_text(
                json.dumps(valid_row) + "\n" + '{"kind":"scenario"',
                encoding="utf-8",
            )

            env = os.environ.copy()
            env.update(
                {
                    "ARTIFACTS_ROOT": str(artifacts_root),
                    "MEASUREMENTS_ROOT": str(tmp_root),
                }
            )
            script = textwrap.dedent(
                f"""\
                try {{
                  const {{ loadRunPageData }} = await import({json.dumps(data_path.as_uri())});
                  loadRunPageData('suite-a', 'run-a');
                  console.log(JSON.stringify({{ ok: true }}));
                }} catch (error) {{
                  console.log(JSON.stringify({{
                    ok: false,
                    name: error.name,
                    message: error.message
                  }}));
                }}
                """
            )
            result = self._run_node(harness_dir=harness_dir, script=script, env=env)

            self.assertEqual(result.returncode, 0, msg=f"{result.stdout}\n{result.stderr}")
            payload = json.loads(result.stdout)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["name"], "ArtifactParseError")
            self.assertIn("transcripts.jsonl", payload["message"])

    def test_load_run_page_data_skips_preview_once_scores_exist(self) -> None:
        with TemporaryDirectory(dir=ROOT / "viewer") as tmp_dir:
            tmp_root = Path(tmp_dir)
            harness_dir = tmp_root / "harness"
            harness_dir.mkdir()
            data_path = self._copy_data_harness(harness_dir)

            artifacts_root = tmp_root / "artifacts" / "results"
            suite_dir = artifacts_root / "suite-a"
            run_dir = suite_dir / "run-a"
            run_dir.mkdir(parents=True, exist_ok=True)

            (suite_dir / "suite.json").write_text(
                json.dumps({"created_at": "2026-04-02T00:00:00Z"}),
                encoding="utf-8",
            )
            (suite_dir / "seeds.jsonl").write_text(
                json.dumps(
                    {
                        "kind": "scenario",
                        "seed_id": "seed-1",
                        "permissible": False,
                        "risk": "risk",
                        "sub_risk": "sub-risk",
                        "definition": "def",
                        "seed": {"title": "Scenario title", "description": "Scenario description"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {"status": "completed", "stages": {"rollout": "completed", "judge": "completed"}}
                ),
                encoding="utf-8",
            )
            valid_row = {
                "kind": "scenario",
                "seed_id": "seed-1",
                "risk": "risk",
                "sub_risk": "sub-risk",
                "permissible": False,
                "stop_reason": "max_turns",
                "events": [],
                "llm_calls": [],
            }
            score_row = {
                "kind": "scenario",
                "seed_id": "seed-1",
                "risk": "risk",
                "sub_risk": "sub-risk",
                "permissible": False,
                "judge_model": "judge-model",
                "judge_status": "ok",
                "verdict": {
                    "dimensions": {"policy_violation": False, "overrefusal": False},
                    "justification": "clear",
                    "node_judgments": [],
                },
            }
            (run_dir / "transcripts.jsonl").write_text(
                json.dumps(valid_row) + "\n",
                encoding="utf-8",
            )
            (run_dir / "scores.jsonl").write_text(
                json.dumps(score_row) + "\n",
                encoding="utf-8",
            )

            env = os.environ.copy()
            env.update(
                {
                    "ARTIFACTS_ROOT": str(artifacts_root),
                    "MEASUREMENTS_ROOT": str(tmp_root),
                }
            )
            script = textwrap.dedent(
                f"""\
                const {{ loadRunPageData }} = await import({json.dumps(data_path.as_uri())});
                const payload = loadRunPageData('suite-a', 'run-a');
                console.log(JSON.stringify({{
                  previewCount: payload.rolloutPreviewRows.length,
                  auditScoreCount: payload.auditScores.length
                }}));
                """
            )
            result = self._run_node(harness_dir=harness_dir, script=script, env=env)

            self.assertEqual(result.returncode, 0, msg=f"{result.stdout}\n{result.stderr}")
            payload = json.loads(result.stdout)
            self.assertEqual(payload["previewCount"], 0)
            self.assertEqual(payload["auditScoreCount"], 1)

    def test_list_suites_marks_scenario_only_scored_suite_as_has_results(self) -> None:
        with TemporaryDirectory(dir=ROOT / "viewer") as tmp_dir:
            tmp_root = Path(tmp_dir)
            harness_dir = tmp_root / "harness"
            harness_dir.mkdir()
            data_path = self._copy_data_harness(harness_dir)

            artifacts_root = tmp_root / "artifacts" / "results"
            suite_dir = artifacts_root / "suite-a"
            run_dir = suite_dir / "run-a"
            run_dir.mkdir(parents=True, exist_ok=True)

            (suite_dir / "suite.json").write_text(
                json.dumps({"created_at": "2026-04-02T00:00:00Z"}),
                encoding="utf-8",
            )
            (suite_dir / "policy.json").write_text(
                json.dumps(
                    {
                        "risk": {"name": "Risk", "definition": "Definition"},
                        "sub_risks": [
                            {
                                "name": "sub-risk",
                                "definition": "def",
                                "examples": [],
                                "permissible": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (suite_dir / "seeds.jsonl").write_text(
                json.dumps(
                    {
                        "kind": "scenario",
                        "seed_id": "seed-1",
                        "permissible": False,
                        "risk": "risk",
                        "sub_risk": "sub-risk",
                        "definition": "def",
                        "seed": {"title": "Scenario title", "description": "Scenario description"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {"status": "completed", "stages": {"rollout": "completed", "judge": "completed"}}
                ),
                encoding="utf-8",
            )
            (run_dir / "scores.jsonl").write_text(
                json.dumps(
                    {
                        "kind": "scenario",
                        "seed_id": "seed-1",
                        "risk": "risk",
                        "sub_risk": "sub-risk",
                        "permissible": False,
                        "judge_model": "judge-model",
                        "judge_status": "ok",
                        "verdict": {
                            "dimensions": {
                                "policy_violation": False,
                                "overrefusal": False,
                            },
                            "justification": "clear",
                            "node_judgments": [],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            env = os.environ.copy()
            env.update(
                {
                    "ARTIFACTS_ROOT": str(artifacts_root),
                    "MEASUREMENTS_ROOT": str(tmp_root),
                }
            )
            script = textwrap.dedent(
                f"""\
                const {{ listSuites }} = await import({json.dumps(data_path.as_uri())});
                console.log(JSON.stringify(listSuites()));
                """
            )
            result = self._run_node(harness_dir=harness_dir, script=script, env=env)

            self.assertEqual(result.returncode, 0, msg=f"{result.stdout}\n{result.stderr}")
            payload = json.loads(result.stdout)
            self.assertEqual(len(payload), 1)
            self.assertEqual(payload[0]["suite_id"], "suite-a")
            self.assertEqual(payload[0]["status"], "has_results")


if __name__ == "__main__":
    unittest.main()
