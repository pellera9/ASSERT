import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from p2m.core.config_model import TargetConfig, ToolsConfig
from p2m.core.model_client import ModelResponse
from p2m.stages.seeds import run as run_stage, run_seeds


class SeedsStageTest(unittest.IsolatedAsyncioTestCase):
    async def test_stage_rejects_removed_validator_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "seeds validators are no longer supported"):
            await run_stage(
                {
                    "suite_root": Path("/tmp/demo-suite"),
                    "config_path": Path("/tmp/config.yaml"),
                    "artifacts_root": Path("/tmp/artifacts"),
                },
                {
                    "validator_model": "azure/gpt-5.4",
                    "prompt": {"model": {"name": "azure/gpt-5.4"}},
                },
            )

    async def test_stage_rejects_removed_generation_context_and_modality_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "seeds.generation_context was renamed to seeds.context"):
            await run_stage(
                {
                    "suite_root": Path("/tmp/demo-suite"),
                    "config_path": Path("/tmp/config.yaml"),
                    "artifacts_root": Path("/tmp/artifacts"),
                },
                {
                    "generation_context": "old",
                    "prompt": {"model": {"name": "azure/gpt-5.4"}},
                },
            )

        with self.assertRaisesRegex(ValueError, "seeds.scenario.modality is no longer supported"):
            await run_stage(
                {
                    "suite_root": Path("/tmp/demo-suite"),
                    "config_path": Path("/tmp/config.yaml"),
                    "artifacts_root": Path("/tmp/artifacts"),
                },
                {
                    "scenario": {"model": {"name": "azure/gpt-5.4"}, "modality": "agentic"},
                },
            )

    async def test_run_seeds_writes_runtime_rows(self) -> None:
        calls: list[str] = []
        prompt_texts: dict[str, str] = {}

        async def fake_generate_structured(model, prompt, *, schema_name, json_schema, options):
            del model, json_schema, options
            calls.append(schema_name)
            prompt_texts[schema_name] = prompt
            if schema_name == "prompt_seeds":
                return ModelResponse(
                    parsed={"seeds": [{"description": "seed one"}]},
                    text="{}",
                    model="azure/gpt-5.4",
                )
            return ModelResponse(
                parsed={"seeds": [{"title": "Scenario A", "description": "Description A"}]},
                text="{}",
                model="azure/gpt-5.4",
            )

        policy_payload = {
            "risk": {"name": "Risk"},
            "sub_risks": [
                {"name": "sub-risk-a", "definition": "definition", "examples": ["example"], "permissible": False},
            ],
        }

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            policy_path = tmp_path / "policy.json"
            seeds_path = tmp_path / "seeds.jsonl"
            policy_path.write_text(json.dumps(policy_payload), encoding="utf-8")

            with patch("p2m.stages.seeds.generate_structured", new=fake_generate_structured):
                result = await run_seeds(
                    policy_path=str(policy_path),
                    save_path=str(seeds_path),
                    context="A coding agent with filesystem and shell tools.",
                    prompt={
                        "model": "azure/gpt-5.4",
                        "budget": 1,
                        "temperature": 0.5,
                        "max_tokens": 1000,
                    },
                    scenario={
                        "model": "azure/gpt-5.4",
                        "budget": 1,
                        "temperature": 0.2,
                        "max_tokens": 1200,
                        "elicitation_strategies": [],
                    },
                    suite_id="suite-1",
                    target=TargetConfig(model="azure/gpt-5.4"),
                    tool_source="runtime",
                )

            rows = [json.loads(line) for line in seeds_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([row["kind"] for row in rows], ["prompt", "scenario"])
        self.assertEqual(rows[0]["seed"]["description"], "seed one")
        self.assertEqual(rows[1]["seed"]["description"], "Description A")
        self.assertEqual(result["seeds_path"], str(seeds_path))
        self.assertEqual(calls, ["prompt_seeds", "scenario_seeds"])
        self.assertIn("You write evaluation seeds: realistic standalone user messages", prompt_texts["prompt_seeds"])
        self.assertIn("A complete standalone user message.", prompt_texts["prompt_seeds"])
        self.assertIn("A coding agent with filesystem and shell tools.", prompt_texts["prompt_seeds"])
        self.assertIn(
            "You write scenario seeds for red-team auditing.",
            prompt_texts["scenario_seeds"],
        )
        self.assertIn("A second-person character brief (400-950 chars).", prompt_texts["scenario_seeds"])
        self.assertIn("A coding agent with filesystem and shell tools.", prompt_texts["scenario_seeds"])
        self.assertNotIn("meta", rows[0])
        self.assertNotIn("meta", rows[1])

    async def test_run_seeds_omits_generated_system_prompts_when_target_prompt_is_fixed(self) -> None:
        async def fake_generate_structured(model, messages, *, schema_name, json_schema, options):
            del model, messages, schema_name, json_schema, options
            return ModelResponse(
                parsed={"seeds": [{"description": "seed one", "system_prompt": "per-seed prompt"}]},
                text="{}",
                model="azure/gpt-5.4",
            )

        policy_payload = {
            "risk": {"name": "Risk"},
            "sub_risks": [
                {"name": "sub-risk-a", "definition": "definition", "examples": ["example"], "permissible": False},
            ],
        }

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            policy_path = tmp_path / "policy.json"
            seeds_path = tmp_path / "seeds.jsonl"
            policy_path.write_text(json.dumps(policy_payload), encoding="utf-8")

            with patch("p2m.stages.seeds.generate_structured", new=fake_generate_structured):
                await run_seeds(
                    policy_path=str(policy_path),
                    save_path=str(seeds_path),
                    context=None,
                    prompt={
                        "model": "azure/gpt-5.4",
                        "budget": 1,
                        "temperature": 0.5,
                        "max_tokens": 1000,
                    },
                    scenario=None,
                    suite_id="suite-1",
                    target=TargetConfig(model="azure/gpt-5.4", system_prompt="fixed prompt"),
                    tool_source="runtime",
                )

            [row] = [json.loads(line) for line in seeds_path.read_text(encoding="utf-8").splitlines()]

        self.assertNotIn("system_prompt", row["seed"])

    async def test_run_seeds_per_seed_requires_simulator_target(self) -> None:
        policy_payload = {
            "risk": {"name": "Risk"},
            "sub_risks": [
                {"name": "sub-risk-a", "definition": "definition", "examples": ["example"], "permissible": False},
            ],
        }

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            policy_path = tmp_path / "policy.json"
            seeds_path = tmp_path / "seeds.jsonl"
            policy_path.write_text(json.dumps(policy_payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "seeds.tool_source=per_seed requires target.tools.simulator"):
                await run_seeds(
                    policy_path=str(policy_path),
                    save_path=str(seeds_path),
                    context=None,
                    prompt={
                        "model": "azure/gpt-5.4",
                        "budget": 1,
                        "temperature": 0.5,
                        "max_tokens": 1000,
                    },
                    scenario=None,
                    suite_id="suite-1",
                    target=TargetConfig(model="azure/gpt-5.4"),
                    tool_source="per_seed",
                )

    async def test_run_seeds_per_seed_emits_tools_and_validates_shape(self) -> None:
        async def fake_generate_structured(model, messages, *, schema_name, json_schema, options):
            del model, messages, schema_name, json_schema, options
            return ModelResponse(
                parsed={
                    "seeds": [
                        {
                            "description": "seed one",
                            "tools": [
                                {
                                    "name": "lookup",
                                    "description": "Fetch account data.",
                                    "parameters": [
                                        {
                                            "name": "account_id",
                                            "type": "string",
                                            "description": "Customer account id.",
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                },
                text="{}",
                model="azure/gpt-5.4",
            )

        policy_payload = {
            "risk": {"name": "Risk"},
            "sub_risks": [
                {"name": "sub-risk-a", "definition": "definition", "examples": ["example"], "permissible": False},
            ],
        }

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            policy_path = tmp_path / "policy.json"
            seeds_path = tmp_path / "seeds.jsonl"
            policy_path.write_text(json.dumps(policy_payload), encoding="utf-8")

            with patch("p2m.stages.seeds.generate_structured", new=fake_generate_structured):
                await run_seeds(
                    policy_path=str(policy_path),
                    save_path=str(seeds_path),
                    context="Exploratory eval across toolsets.",
                    prompt={
                        "model": "azure/gpt-5.4",
                        "budget": 1,
                        "temperature": 0.5,
                        "max_tokens": 1000,
                    },
                    scenario=None,
                    suite_id="suite-1",
                    target=TargetConfig(
                        model="azure/gpt-5.4",
                        tools=ToolsConfig(simulator="azure/gpt-5.4-mini"),
                    ),
                    tool_source="per_seed",
                )

            [row] = [json.loads(line) for line in seeds_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(row["seed"]["tools"][0]["name"], "lookup")

    async def test_run_seeds_per_seed_rejects_invalid_generated_tool_payloads(self) -> None:
        async def fake_generate_structured(model, messages, *, schema_name, json_schema, options):
            del model, messages, schema_name, json_schema, options
            return ModelResponse(
                parsed={"seeds": [{"description": "seed one", "tools": [{}]}]},
                text="{}",
                model="azure/gpt-5.4",
            )

        policy_payload = {
            "risk": {"name": "Risk"},
            "sub_risks": [
                {"name": "sub-risk-a", "definition": "definition", "examples": ["example"], "permissible": False},
            ],
        }

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            policy_path = tmp_path / "policy.json"
            seeds_path = tmp_path / "seeds.jsonl"
            policy_path.write_text(json.dumps(policy_payload), encoding="utf-8")

            with patch("p2m.stages.seeds.generate_structured", new=fake_generate_structured):
                with self.assertRaisesRegex(ValueError, "generated seed contains invalid tool definitions"):
                    await run_seeds(
                        policy_path=str(policy_path),
                        save_path=str(seeds_path),
                        context=None,
                        prompt={
                            "model": "azure/gpt-5.4",
                            "budget": 1,
                            "temperature": 0.5,
                            "max_tokens": 1000,
                        },
                        scenario=None,
                        suite_id="suite-1",
                        target=TargetConfig(
                            model="azure/gpt-5.4",
                            tools=ToolsConfig(simulator="azure/gpt-5.4-mini"),
                        ),
                        tool_source="per_seed",
                    )


if __name__ == "__main__":
    unittest.main()
