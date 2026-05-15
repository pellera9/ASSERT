import unittest
from types import SimpleNamespace
from unittest.mock import patch

from p2m.core import model_client


class ModelClientTest(unittest.IsolatedAsyncioTestCase):
    async def test_generate_uses_acompletion_and_normalizes_response(self) -> None:
        captured: dict[str, object] = {}

        async def fake_acompletion(**kwargs):
            captured.update(kwargs)
            return {
                "id": "resp-chat-1",
                "model": "openai/gpt-5-mini",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "hello world",
                            "tool_calls": None,
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
            }

        fake_litellm = SimpleNamespace(acompletion=fake_acompletion)
        options = model_client.GenerateOptions(temperature=0.2, max_tokens=64)

        with patch.object(model_client, "_get_litellm_module", return_value=fake_litellm):
            response = await model_client.generate(
                "openai/gpt-5-mini",
                "say hi",
                options,
            )

        self.assertEqual(
            captured["messages"],
            [{"role": "user", "content": "say hi"}],
        )
        self.assertEqual(captured["temperature"], 0.2)
        self.assertEqual(captured["max_tokens"], 64)
        self.assertEqual(response.text, "hello world")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.usage.total_tokens, 18)
        self.assertEqual(response.api_mode, "chat_completion")
        self.assertEqual(response.request_payload["model"], "openai/gpt-5-mini")
        self.assertEqual(response.request_payload["messages"], [{"role": "user", "content": "say hi"}])

    async def test_generate_structured_adds_json_schema_response_format(self) -> None:
        captured: dict[str, object] = {}

        async def fake_acompletion(**kwargs):
            captured.update(kwargs)
            return {
                "id": "resp-structured-1",
                "model": "openai/gpt-5-mini",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '{"verdict": "pass"}',
                        },
                    }
                ],
            }

        fake_litellm = SimpleNamespace(acompletion=fake_acompletion)
        schema = {
            "type": "object",
            "properties": {
                "verdict": {"type": "string"},
            },
            "required": ["verdict"],
            "additionalProperties": False,
        }

        with patch.object(model_client, "_get_litellm_module", return_value=fake_litellm):
            response = await model_client.generate_structured(
                "openai/gpt-5-mini",
                [{"role": "user", "content": "judge this"}],
                schema_name="judge_output",
                json_schema=schema,
            )

        response_format = captured["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["name"], "judge_output")
        self.assertEqual(response_format["json_schema"]["schema"], schema)
        self.assertEqual(response.parsed, {"verdict": "pass"})

    async def test_generate_structured_with_web_search_rejects_gemini(self) -> None:
        schema = {
            "type": "object",
            "properties": {"verdict": {"type": "string"}},
            "required": ["verdict"],
            "additionalProperties": False,
        }

        with patch.object(model_client, "_get_litellm_module") as get_litellm:
            with self.assertRaisesRegex(ValueError, "web_search.*gemini/gemini-2.5-flash"):
                await model_client.generate_structured(
                    "gemini/gemini-2.5-flash",
                    "research this",
                    schema_name="judge_output",
                    json_schema=schema,
                    options=model_client.GenerateOptions(web_search=True),
                )

        get_litellm.assert_not_called()

    async def test_generate_with_web_search_rejects_non_openai_provider(self) -> None:
        with patch.object(model_client, "_get_litellm_module") as get_litellm:
            with self.assertRaisesRegex(ValueError, "Disable web_search"):
                await model_client.generate(
                    "anthropic/claude-sonnet-4-20250514",
                    "research this",
                    options=model_client.GenerateOptions(web_search=True),
                )

        get_litellm.assert_not_called()

    async def test_generate_structured_with_web_search_uses_responses_api(self) -> None:
        captured: dict[str, object] = {}

        async def fake_aresponses(**kwargs):
            captured.update(kwargs)
            return {
                "id": "resp-structured-search-1",
                "model": "openai/gpt-5-mini",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"verdict": "pass"}',
                            }
                        ],
                    }
                ],
            }

        fake_litellm = SimpleNamespace(aresponses=fake_aresponses)
        schema = {
            "type": "object",
            "properties": {
                "verdict": {"type": "string"},
            },
            "required": ["verdict"],
            "additionalProperties": False,
        }

        with patch.object(model_client, "_get_litellm_module", return_value=fake_litellm):
            response = await model_client.generate_structured(
                "openai/gpt-5-mini",
                "research this",
                schema_name="judge_output",
                json_schema=schema,
                options=model_client.GenerateOptions(web_search=True, reasoning_effort="high"),
            )

        self.assertEqual(captured["input"], "research this")
        self.assertEqual(captured["reasoning_effort"], "high")
        self.assertEqual(captured["tools"], [{"type": "web_search_preview"}])
        self.assertEqual(captured["text"]["format"]["type"], "json_schema")
        self.assertEqual(captured["text"]["format"]["name"], "judge_output")
        self.assertEqual(captured["text"]["format"]["schema"], schema)
        self.assertEqual(response.parsed, {"verdict": "pass"})

    async def test_generate_with_tools_normalizes_tool_calls(self) -> None:
        async def fake_acompletion(**_kwargs):
            return {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "send_message",
                                        "arguments": '{"message": "hello"}',
                                    },
                                }
                            ],
                        },
                    }
                ]
            }

        fake_litellm = SimpleNamespace(acompletion=fake_acompletion)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "send_message",
                    "description": "Send a message",
                    "parameters": {"type": "object"},
                },
            }
        ]

        with patch.object(model_client, "_get_litellm_module", return_value=fake_litellm):
            response = await model_client.generate_with_tools(
                "openai/gpt-5-mini",
                "use a tool",
                tools=tools,
                options=model_client.GenerateOptions(tool_choice="auto"),
            )

        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].call_id, "call_1")
        self.assertEqual(response.tool_calls[0].name, "send_message")
        self.assertEqual(response.tool_calls[0].arguments, {"message": "hello"})
        self.assertEqual(response.api_mode, "chat_completion")
        self.assertEqual(response.request_payload["tools"], tools)
        self.assertEqual(response.request_payload["tool_choice"], "auto")

    async def test_generate_with_web_search_falls_back_to_sync_responses(self) -> None:
        captured: dict[str, object] = {}

        def fake_responses(**kwargs):
            captured.update(kwargs)
            return {
                "id": "resp-search-1",
                "model": "openai/gpt-5-mini",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "research result",
                            "reasoning_content": "internal reasoning",
                        },
                    }
                ],
            }

        fake_litellm = SimpleNamespace(responses=fake_responses)
        options = model_client.GenerateOptions(
            web_search=True,
            reasoning_effort="high",
            max_output_tokens=2048,
        )

        with patch.object(model_client, "_get_litellm_module", return_value=fake_litellm):
            response = await model_client.generate(
                "openai/gpt-5-mini",
                "research this",
                options,
            )

        self.assertEqual(captured["input"], "research this")
        self.assertEqual(captured["max_output_tokens"], 2048)
        self.assertEqual(captured["reasoning_effort"], "high")
        self.assertEqual(captured["tools"], [{"type": "web_search_preview"}])
        self.assertEqual(response.text, "research result")
        self.assertEqual(response.reasoning, "internal reasoning")


if __name__ == "__main__":
    unittest.main()
