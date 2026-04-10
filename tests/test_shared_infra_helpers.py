import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from p2m.core.io import (
    load_seeds,
    resolve_path,
    write_jsonl,
)
import p2m.core.judge as judge_module
from p2m.core.judge import (
    build_judge_contract,
    multi_judge,
    resolve_dimensions,
    run_judge,
    run_transcript_judge,
)
from p2m.core.model_client import ModelResponse
from p2m.core.transcript import AddMessageEdit, Message, Transcript, TranscriptEvent, TranscriptMetadata
from p2m.core.tools import build_target_tools


class SharedInfraHelpersTest(unittest.IsolatedAsyncioTestCase):
    def _transcript_with_assistant_message(self, content: str) -> tuple[Transcript, dict[str, str]]:
        transcript = Transcript(
            metadata=TranscriptMetadata(
                kind="scenario",
                seed_id="seed-1",
                risk="risk",
                sub_risk="sub-risk",
                permissible=False,
                target="target",
                auditor_model="auditor",
            )
        )
        transcript.add_event(
            TranscriptEvent(
                view=["target", "combined"],
                actor="target",
                edit=AddMessageEdit(message=Message(role="assistant", content=content)),
            )
        )
        _, index_to_message_id = transcript.format_transcript_xml(
            "target",
            skip_system=False,
        )
        return transcript, index_to_message_id

    def test_build_target_tools_returns_openai_tool_dicts(self) -> None:
        tools = build_target_tools(
            [
                {
                    "name": "lookup_record",
                    "description": "Look up a record.",
                    "parameters": [
                        {"name": "record_id", "type": "string", "description": "Record ID"},
                        {"name": "fields", "type": "array", "description": "Fields to fetch"},
                    ],
                }
            ]
        )

        self.assertEqual(len(tools), 1)
        tool = tools[0]
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["function"]["name"], "lookup_record")
        self.assertEqual(tool["function"]["parameters"]["required"], ["record_id", "fields"])
        self.assertEqual(
            tool["function"]["parameters"]["properties"]["fields"]["items"],
            {"type": "string"},
        )

    def test_utils_shim_write_jsonl_and_load_seeds_round_trip(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "seeds.jsonl"
            write_jsonl(path, [{"prompt": "one"}, {"prompt": "two"}])

            rows = load_seeds(path)
            self.assertEqual(rows, [{"prompt": "one"}, {"prompt": "two"}])

    def test_utils_shim_load_seeds_skips_malformed_lines_when_not_strict(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "seeds.jsonl"
            path.write_text('{"prompt":"ok"}\nnot-json\n', encoding="utf-8")

            rows = load_seeds(path, strict=False)
            self.assertEqual(rows, [{"prompt": "ok"}])

            with self.assertRaises(ValueError):
                load_seeds(path, strict=True)

    def test_utils_shim_resolve_dimensions_uses_bundled_dimension_data(self) -> None:
        dims = resolve_dimensions(["overrefusal"])
        self.assertEqual(len(dims), 1)
        self.assertEqual(dims[0]["name"], "overrefusal")
        self.assertIn("description", dims[0])
        self.assertIn("rubric", dims[0])

    def test_resolve_dimensions_rejects_non_event_dimension_kinds(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            dimensions_path = Path(tmp_dir) / "judge_dimensions.yaml"
            dimensions_path.write_text(
                "bad_metric:\n  kind: rubric\n  description: Bad\n  rubric: bad\n",
                encoding="utf-8",
            )
            with patch.object(judge_module, "DIMENSIONS_PATH", dimensions_path):
                with self.assertRaisesRegex(ValueError, "must declare kind: 'event'"):
                    resolve_dimensions(["bad_metric"])

    def test_utils_shim_resolve_path_preserves_absolute_paths(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "artifact.json"
            path.write_text(json.dumps({"ok": True}), encoding="utf-8")
            self.assertEqual(resolve_path(path), path)

    async def test_multi_judge_uses_structured_path_when_schema_present(self) -> None:
        class ResponseSchema:
            name = "judge_schema"
            json_schema = {
                "type": "object",
                "properties": {
                    "dimensions": {
                        "type": "object",
                        "properties": {
                            "policy_violation": {"type": "boolean"},
                            "overrefusal": {"type": "boolean"},
                        },
                        "required": ["policy_violation", "overrefusal"],
                        "additionalProperties": False,
                    },
                    "dimension_justifications": {
                        "type": "object",
                        "properties": {
                            "policy_violation": {"type": "string"},
                            "overrefusal": {"type": "string"},
                        },
                        "required": ["policy_violation", "overrefusal"],
                        "additionalProperties": False,
                    },
                    "node_judgments": {"type": "array"},
                },
                "required": ["dimensions", "dimension_justifications", "node_judgments"],
                "additionalProperties": False,
            }

        async def fake_generate_structured(model, messages, *, schema_name, json_schema, options):
            self.assertEqual(schema_name, "judge_schema")
            self.assertEqual(messages[0].role, "system")
            self.assertEqual(messages[1].role, "user")
            return ModelResponse(
                parsed={
                    "dimensions": {"policy_violation": False, "overrefusal": False},
                    "dimension_justifications": {
                        "policy_violation": "good [1]",
                        "overrefusal": "good [1]",
                    },
                    "node_judgments": [],
                },
                text='{"dimensions":{"policy_violation":false,"overrefusal":false},"dimension_justifications":{"policy_violation":"good [1]","overrefusal":"good [1]"},"node_judgments":[]}',
                model=model,
            )

        with patch("p2m.core.judge.generate_structured", new=fake_generate_structured):
            result = await multi_judge(
                judge_model="azure/gpt-5.4",
                system_prompt="system",
                user_message="user",
                score_keys=["policy_violation"],
                judge_n=1,
                judge_temperature=0.0,
                judge_max_tokens=200,
                response_schema=ResponseSchema(),
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["verdict"]["dimensions"]["policy_violation"], False)

    async def test_multi_judge_uses_raw_json_when_structured_parsed_dict_is_incomplete(self) -> None:
        class ResponseSchema:
            name = "judge_schema"
            json_schema = {
                "type": "object",
                "properties": {
                    "dimensions": {
                        "type": "object",
                        "properties": {
                            "policy_violation": {"type": "boolean"},
                            "overrefusal": {"type": "boolean"},
                        },
                        "required": ["policy_violation", "overrefusal"],
                        "additionalProperties": False,
                    },
                    "dimension_justifications": {
                        "type": "object",
                        "properties": {
                            "policy_violation": {"type": "string"},
                            "overrefusal": {"type": "string"},
                        },
                        "required": ["policy_violation", "overrefusal"],
                        "additionalProperties": False,
                    },
                    "node_judgments": {"type": "array"},
                },
                "required": ["dimensions", "dimension_justifications", "node_judgments"],
                "additionalProperties": False,
            }

        async def fake_generate_structured(model, messages, *, schema_name, json_schema, options):
            self.assertEqual(schema_name, "judge_schema")
            self.assertEqual(messages[0].role, "system")
            self.assertEqual(messages[1].role, "user")
            return ModelResponse(
                parsed={"dimensions": {"policy_violation": False, "overrefusal": False}},
                text='{"dimensions":{"policy_violation":false,"overrefusal":false},"dimension_justifications":{"policy_violation":"good [1]","overrefusal":"good [1]"},"node_judgments":[]}',
                model=model,
            )

        with patch("p2m.core.judge.generate_structured", new=fake_generate_structured):
            result = await multi_judge(
                judge_model="github_copilot/claude-opus-4.6",
                system_prompt="system",
                user_message="user",
                score_keys=["policy_violation"],
                judge_n=1,
                judge_temperature=0.0,
                judge_max_tokens=200,
                response_schema=ResponseSchema(),
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["verdict"]["dimensions"]["policy_violation"], False)

    async def test_multi_judge_retries_without_schema_when_structured_output_is_freeform(self) -> None:
        class ResponseSchema:
            name = "judge_schema"
            json_schema = {
                "type": "object",
                "properties": {
                    "dimensions": {
                        "type": "object",
                        "properties": {
                            "policy_violation": {"type": "boolean"},
                            "overrefusal": {"type": "boolean"},
                        },
                        "required": ["policy_violation", "overrefusal"],
                        "additionalProperties": False,
                    },
                    "dimension_justifications": {
                        "type": "object",
                        "properties": {
                            "policy_violation": {"type": "string"},
                            "overrefusal": {"type": "string"},
                        },
                        "required": ["policy_violation", "overrefusal"],
                        "additionalProperties": False,
                    },
                    "node_judgments": {"type": "array"},
                },
                "required": ["dimensions", "dimension_justifications", "node_judgments"],
                "additionalProperties": False,
            }

        async def fake_generate_structured(model, messages, *, schema_name, json_schema, options):
            self.assertEqual(schema_name, "judge_schema")
            self.assertEqual(messages[0].role, "system")
            self.assertEqual(messages[1].role, "user")
            return ModelResponse(
                parsed=None,
                text="Let me think through the policy carefully before deciding.",
                model=model,
            )

        async def fake_generate(model, messages, *, options):
            self.assertEqual(messages[0].role, "system")
            self.assertEqual(messages[1].role, "user")
            return ModelResponse(
                text='{"dimensions":{"policy_violation":false,"overrefusal":false},"dimension_justifications":{"policy_violation":"good [1]","overrefusal":"good [1]"},"node_judgments":[]}',
                model=model,
            )

        with (
            patch("p2m.core.judge.generate_structured", new=fake_generate_structured),
            patch("p2m.core.judge.generate", new=fake_generate),
        ):
            result = await multi_judge(
                judge_model="github_copilot/claude-opus-4.6",
                system_prompt="system",
                user_message="user",
                score_keys=["policy_violation"],
                judge_n=1,
                judge_temperature=0.0,
                judge_max_tokens=200,
                response_schema=ResponseSchema(),
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["verdict"]["dimensions"]["policy_violation"], False)

    async def test_multi_judge_does_not_retry_without_schema_for_non_copilot_models(self) -> None:
        class ResponseSchema:
            name = "judge_schema"
            json_schema = {
                "type": "object",
                "properties": {
                    "dimensions": {
                        "type": "object",
                        "properties": {
                            "policy_violation": {"type": "boolean"},
                            "overrefusal": {"type": "boolean"},
                        },
                        "required": ["policy_violation", "overrefusal"],
                        "additionalProperties": False,
                    },
                    "dimension_justifications": {
                        "type": "object",
                        "properties": {
                            "policy_violation": {"type": "string"},
                            "overrefusal": {"type": "string"},
                        },
                        "required": ["policy_violation", "overrefusal"],
                        "additionalProperties": False,
                    },
                    "node_judgments": {"type": "array"},
                },
                "required": ["dimensions", "dimension_justifications", "node_judgments"],
                "additionalProperties": False,
            }

        async def fake_generate_structured(model, messages, *, schema_name, json_schema, options):
            self.assertEqual(model, "azure/gpt-5.4")
            return ModelResponse(
                parsed=None,
                text="Let me think through the policy carefully before deciding.",
                model=model,
            )

        async def fake_generate(model, messages, *, options):
            raise AssertionError("non-copilot models should not retry without schema")

        with (
            patch("p2m.core.judge.generate_structured", new=fake_generate_structured),
            patch("p2m.core.judge.generate", new=fake_generate),
        ):
            result = await multi_judge(
                judge_model="azure/gpt-5.4",
                system_prompt="system",
                user_message="user",
                score_keys=["policy_violation"],
                judge_n=1,
                judge_temperature=0.0,
                judge_max_tokens=200,
                response_schema=ResponseSchema(),
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["failures"], 1)
        self.assertIsNone(result["verdict"])

    def test_build_judge_contract_respects_citation_style(self) -> None:
        response_contract = build_judge_contract(
            template="Judge {{policy_json}} {{dimensions_section}} {{output_schema}}",
            policy_raw={"sub_risks": []},
            judge_dimensions=["overrefusal"],
            citation_style="response",
            schema_name="response_judgment",
        )
        turn_contract = build_judge_contract(
            template="Judge {{policy_json}} {{dimensions_section}} {{output_schema}}",
            policy_raw={"sub_risks": []},
            judge_dimensions=["overrefusal"],
            citation_style="turn",
            schema_name="turn_judgment",
        )
        xml_contract = build_judge_contract(
            template="Judge {{policy_json}} {{dimensions_section}} {{output_schema}}",
            policy_raw={"sub_risks": []},
            judge_dimensions=["overrefusal"],
            citation_style="xml",
            schema_name="xml_judgment",
        )

        response_required = response_contract["response_schema"]["json_schema"]["properties"]["citations"]["items"]["required"]
        turn_required = turn_contract["response_schema"]["json_schema"]["properties"]["citations"]["items"]["required"]

        self.assertEqual(response_contract["response_schema"]["name"], "response_judgment")
        self.assertEqual(turn_contract["response_schema"]["name"], "turn_judgment")
        self.assertEqual(xml_contract["response_schema"]["name"], "xml_judgment")
        self.assertIn("policy_violation", response_contract["score_keys"])
        self.assertIn("overrefusal", response_contract["score_keys"])
        self.assertIn("dimension_justifications", response_contract["response_schema"]["json_schema"]["required"])
        self.assertNotIn("turn", response_required)
        self.assertIn("turn", turn_required)
        self.assertEqual(xml_contract["response_schema"]["json_schema"]["properties"]["highlights"]["type"], "string")
        self.assertIn('"sub_risks": []', response_contract["system_prompt"])

    def test_build_judge_contract_sets_exact_node_count(self) -> None:
        contract = build_judge_contract(
            template="Judge {{policy_json}} {{dimensions_section}} {{output_schema}}",
            policy_raw={"sub_risks": [{"name": "a"}, {"name": "b"}, {"name": "c"}]},
            judge_dimensions=["overrefusal"],
            citation_style="xml",
            schema_name="xml_judgment",
        )

        node_schema = contract["response_schema"]["json_schema"]["properties"]["node_judgments"]
        self.assertEqual(node_schema["maxItems"], 3)
        self.assertEqual(node_schema["items"]["properties"]["node_index"]["maximum"], 2)

    async def test_run_judge_returns_normalized_success_payload(self) -> None:
        async def fake_multi_judge(**kwargs):
            self.assertEqual(kwargs["judge_model"], "azure/gpt-5.4")
            return {
                "verdict": {
                    "dimensions": {"policy_violation": False, "overrefusal": False},
                    "dimension_justifications": {
                        "policy_violation": "good [1]",
                        "overrefusal": "good [1]",
                    },
                    "node_judgments": [],
                },
                "raw": '{"dimensions":{"policy_violation":false,"overrefusal":false},"dimension_justifications":{"policy_violation":"good [1]","overrefusal":"good [1]"},"node_judgments":[]}',
                "multi_judge": {
                    "n": 2,
                    "n_failed": 0,
                    "votes": {"policy_violation": [False, False], "overrefusal": [False, False]},
                },
                "success": True,
                "failures": 0,
            }

        with patch("p2m.core.judge.multi_judge", new=fake_multi_judge):
            result = await run_judge(
                judge_model="azure/gpt-5.4",
                system_prompt="system",
                user_message="user",
                score_keys=["policy_violation"],
                judge_n=2,
                judge_temperature=0.4,
                judge_max_tokens=200,
                response_schema={"name": "judge_schema", "json_schema": {"type": "object"}},
            )

        self.assertEqual(result["judge_status"], "ok")
        self.assertIsNone(result["judge_error"])
        self.assertEqual(result["verdict"]["dimensions"]["policy_violation"], False)
        self.assertEqual(result["verdict"]["dimension_justifications"]["policy_violation"], "good [1]")
        self.assertEqual(result["score_values"]["policy_violation"], 0.0)
        self.assertEqual(result["score_meta"]["policy_violation_raw"], False)
        self.assertEqual(result["multi_judge"]["n"], 2)
        self.assertEqual(result["multi_judge"]["n_failed"], 0)

    async def test_run_judge_returns_normalized_failure_payload(self) -> None:
        async def fake_multi_judge(**kwargs):
            self.assertEqual(kwargs["score_keys"], ["policy_violation"])
            return {
                "verdict": None,
                "raw": "bad output",
                "multi_judge": None,
                "success": False,
                "failures": 1,
            }

        with patch("p2m.core.judge.multi_judge", new=fake_multi_judge):
            result = await run_judge(
                judge_model="azure/gpt-5.4",
                system_prompt="system",
                user_message="user",
                score_keys=["policy_violation"],
            )

        self.assertEqual(result["judge_status"], "judge_failed")
        self.assertEqual(result["judge_error"], "judge_failed")
        self.assertEqual(result["verdict"], {"error": "judge_failed"})
        self.assertEqual(result["score_values"], {"policy_violation": 0.0})
        self.assertEqual(result["score_meta"], {})
        self.assertIsNone(result["multi_judge"])
        self.assertEqual(result["raw"], "bad output")

    async def test_run_transcript_judge_normalizes_sparse_nodes(self) -> None:
        transcript, index_to_message_id = self._transcript_with_assistant_message(
            "Stay calm and call the nurse if this keeps getting worse."
        )

        async def fake_run_judge_attempts(*args, **kwargs):
            return (
                [
                    {
                        "dimensions": {
                            "policy_violation": False,
                            "overrefusal": False,
                        },
                        "dimension_justifications": {
                            "policy_violation": "Allowed supportive guidance [1]",
                            "overrefusal": "The assistant helped instead of refusing [1]",
                        },
                        "node_judgments": [
                            {
                                "node_index": 1,
                                "violated": False,
                                "confidence": "high",
                                "reasoning": "Allowed supportive guidance [1]",
                            }
                        ],
                        "highlights": '1. <cite id="1" description="Supportive guidance">Stay calm and call the nurse if this keeps getting worse.</cite>',
                    }
                ],
                ['{"judge":"ok"}'],
                0,
            )

        with patch("p2m.core.judge._run_judge_attempts", new=fake_run_judge_attempts):
            result = await run_transcript_judge(
                judge_model="azure/gpt-5.4",
                system_prompt="system",
                user_message="user",
                transcript=transcript,
                index_to_message_id=index_to_message_id,
                score_keys=["policy_violation", "overrefusal"],
                policy_raw={"sub_risks": [{"name": "a"}, {"name": "b"}]},
            )

        self.assertEqual(result["judge_status"], "ok")
        self.assertEqual(result["verdict"]["justification"], "Allowed supportive guidance [1]")
        self.assertEqual(result["verdict"]["node_judgments"][0]["node_name"], "a")
        self.assertFalse(result["verdict"]["node_judgments"][0]["relevant"])
        self.assertEqual(result["verdict"]["node_judgments"][1]["node_name"], "b")
        self.assertTrue(result["verdict"]["node_judgments"][1]["relevant"])
        self.assertEqual(result["verdict"]["node_judgments"][1]["evidence_turns"], [1])
        self.assertIn("citations", result["verdict"])

    async def test_run_transcript_judge_returns_first_normalization_error(self) -> None:
        transcript, index_to_message_id = self._transcript_with_assistant_message("Hello")

        async def fake_run_judge_attempts(*args, **kwargs):
            return (
                [
                    {
                        "dimensions": {
                            "policy_violation": False,
                            "overrefusal": False,
                        },
                        "dimension_justifications": {
                            "policy_violation": "safe [1]",
                            "overrefusal": "safe [1]",
                        },
                        "node_judgments": [],
                        "highlights": "",
                    }
                ],
                ['{"judge":"bad"}'],
                0,
            )

        with patch("p2m.core.judge._run_judge_attempts", new=fake_run_judge_attempts):
            result = await run_transcript_judge(
                judge_model="azure/gpt-5.4",
                system_prompt="system",
                user_message="user",
                transcript=transcript,
                index_to_message_id=index_to_message_id,
                score_keys=["policy_violation", "overrefusal"],
                policy_raw={"sub_risks": []},
            )

        self.assertEqual(result["judge_status"], "judge_failed")
        self.assertEqual(result["judge_error"], "missing_citations")
        self.assertEqual(result["verdict"], {"error": "judge_failed"})
        self.assertEqual(result["score_values"], {"policy_violation": 0.0, "overrefusal": 0.0})
        self.assertEqual(result["raw"], '{"judge":"bad"}')


if __name__ == "__main__":
    unittest.main()
