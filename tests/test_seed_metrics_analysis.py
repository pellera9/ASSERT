import unittest

from p2m.analysis.seed_metrics import _prompt_seed_text


class SeedMetricsAnalysisTest(unittest.TestCase):
    def test_prompt_seed_text_ignores_scenario_rows(self) -> None:
        self.assertEqual(
            _prompt_seed_text({"kind": "prompt", "seed": {"description": "seed"}}),
            "seed",
        )
        self.assertIsNone(
            _prompt_seed_text({"kind": "scenario", "seed": {"description": "hidden brief"}})
        )


if __name__ == "__main__":
    unittest.main()
