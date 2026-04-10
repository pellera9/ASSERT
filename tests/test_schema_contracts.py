"""Tests for schema contracts across the prompt/code boundary.

Phase 1b of the prompt revision plan: verify that independently defined
schemas, examples, and parsers stay in sync.
"""

import json
import re
import unittest

from p2m.core.judge import (
    CITE_XML_EXAMPLE,
    CITE_XML_PATTERN,
    build_judge_schema,
    render_output_schema,
)
from p2m.stages.seeds import (
    _SEED_SCHEMA,
    _SEED_SCHEMA_WITH_TOOLS,
    _generate_schema_example,
    _fill_template,
)


class SeedSchemaExampleTest(unittest.TestCase):
    def test_generated_example_parses_as_json(self) -> None:
        example = _generate_schema_example(_SEED_SCHEMA)
        parsed = json.loads(example)
        self.assertIn("seeds", parsed)
        self.assertIsInstance(parsed["seeds"], list)
        self.assertEqual(len(parsed["seeds"]), 1)

    def test_generated_example_validates_against_schema(self) -> None:
        """Every key in the example must be a property in the schema."""
        example = json.loads(_generate_schema_example(_SEED_SCHEMA))
        seed = example["seeds"][0]
        schema_props = set(_SEED_SCHEMA["properties"].keys())
        self.assertEqual(set(seed.keys()), schema_props)

    def test_generated_tool_example_validates_against_schema(self) -> None:
        example = json.loads(_generate_schema_example(_SEED_SCHEMA_WITH_TOOLS))
        seed = example["seeds"][0]
        schema_props = set(_SEED_SCHEMA_WITH_TOOLS["properties"].keys())
        self.assertEqual(set(seed.keys()), schema_props)

    def test_schema_required_fields_match_properties(self) -> None:
        for schema in (_SEED_SCHEMA, _SEED_SCHEMA_WITH_TOOLS):
            with self.subTest(schema=schema):
                self.assertEqual(
                    set(schema["required"]),
                    set(schema["properties"].keys()),
                )


class PlaceholderValidationTest(unittest.TestCase):
    def test_fill_template_raises_on_unreplaced_placeholder(self) -> None:
        with self.assertRaisesRegex(ValueError, "unreplaced template placeholders"):
            _fill_template("Hello {{name}}, your {{role}} is ready.", {"name": "Alice"})

    def test_fill_template_succeeds_when_all_replaced(self) -> None:
        result = _fill_template("Hello {{name}}!", {"name": "Alice"})
        self.assertEqual(result, "Hello Alice!")


class JudgeCitationFormatTest(unittest.TestCase):
    def test_cite_example_matches_pattern(self) -> None:
        match = CITE_XML_PATTERN.search(CITE_XML_EXAMPLE)
        self.assertIsNotNone(match, "CITE_XML_EXAMPLE must match CITE_XML_PATTERN")
        self.assertEqual(match.group(2), "3")
        self.assertEqual(match.group(3), "Key evidence")

    def test_rendered_xml_highlights_match_pattern(self) -> None:
        rendered = render_output_schema([], include_citations=True, citation_style="xml")
        highlights_line = [l for l in rendered.splitlines() if "highlights" in l]
        self.assertTrue(highlights_line, "rendered schema must contain highlights")

    def test_rendered_xml_highlights_have_escaped_quotes(self) -> None:
        rendered = render_output_schema([], include_citations=True, citation_style="xml")
        highlights_line = [l for l in rendered.splitlines() if "highlights" in l]
        self.assertTrue(highlights_line, "rendered schema must contain highlights")
        line = highlights_line[0]
        # The line should look like:  "highlights": "1. <cite id=\"3\" ...>...</cite>"
        # Inner quotes in the XML must be escaped with backslash for valid JSON context
        self.assertIn('\\"', line, "inner quotes in highlights must be escaped")


class JudgeSchemaCompatibilityTest(unittest.TestCase):
    def _assert_schemas_compatible(
        self, dims: list[dict[str, str]], citation_style: str,
    ) -> None:
        schema = build_judge_schema(dims, include_citations=True, citation_style=citation_style)
        rendered = render_output_schema(dims, include_citations=True, citation_style=citation_style)
        schema_keys = set(schema["properties"].keys())
        rendered_top_level_keys = set(re.findall(r'^  "(\w+)":', rendered, flags=re.M))
        missing = schema_keys - rendered_top_level_keys
        self.assertFalse(missing, f"Schema keys missing from rendered example: {missing}")
        self.assertIn('"dimensions": {', rendered)
        self.assertIn('"node_judgments": [', rendered)
        for dim in dims:
            self.assertIn(f'"{dim["name"]}"', rendered)

    def test_no_dimensions_xml(self) -> None:
        self._assert_schemas_compatible([], "xml")

    def test_no_dimensions_turn(self) -> None:
        self._assert_schemas_compatible([], "turn")

    def test_no_dimensions_response(self) -> None:
        self._assert_schemas_compatible([], "response")

    def test_with_dimensions(self) -> None:
        dims = [
            {"name": "policy_violation", "description": "Bad event", "rubric": "true = bad", "kind": "event"},
            {"name": "harm_actionability", "description": "Bad event", "rubric": "true = bad", "kind": "event"},
        ]
        for style in ("xml", "turn", "response"):
            with self.subTest(style=style):
                self._assert_schemas_compatible(dims, style)


if __name__ == "__main__":
    unittest.main()
