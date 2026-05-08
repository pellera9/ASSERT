import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from p2m.core.artifact_cache import (
    activate_artifact_plan,
    finalize_artifact_plan,
    hash_payload,
    prepare_artifact_plan,
)


class ArtifactCacheTest(unittest.TestCase):
    def _ctx(self, root: Path) -> dict:
        config_path = root / "config.yaml"
        config_path.write_text("suite: suite-a\n", encoding="utf-8")
        suite_root = root / "results" / "suite-a"
        suite_root.mkdir(parents=True, exist_ok=True)
        return {
            "config_path": config_path,
            "artifacts_root": root / "artifacts",
            "suite_root": suite_root,
            "concept_name": "travel_planner_eval",
            "concept": "Travel planner must produce grounded itineraries.",
            "context": "Travel planner with flight and hotel tools.",
            "artifact_versions": {},
        }

    def test_hash_payload_is_stable_across_dict_key_order(self) -> None:
        self.assertEqual(
            hash_payload({"b": [2, {"d": 4, "c": 3}], "a": 1}),
            hash_payload({"a": 1, "b": [2, {"c": 3, "d": 4}]}),
        )

    def test_prepare_reuses_latest_matching_artifact(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            ctx = self._ctx(root)
            raw_cfg = {"model": {"name": "azure/gpt-5.4"}, "behavior_count": 5}

            first = prepare_artifact_plan(
                ctx=ctx,
                stage_name="policy",
                raw_cfg=raw_cfg,
                forced=False,
            )
            activate_artifact_plan(ctx, first)
            first.output_paths["policy"].parent.mkdir(parents=True, exist_ok=True)
            first.output_paths["policy"].write_text('{"behaviors":[]}', encoding="utf-8")
            first.output_paths["systematization"].write_text("{}", encoding="utf-8")
            finalize_artifact_plan(ctx, first)

            second_ctx = self._ctx(root)
            second = prepare_artifact_plan(
                ctx=second_ctx,
                stage_name="policy",
                raw_cfg=raw_cfg,
                forced=False,
            )

            self.assertTrue(second.reused)
            self.assertEqual(second.version, "v0001")

    def test_hash_mismatch_allocates_next_version(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            ctx = self._ctx(root)
            raw_cfg = {"model": {"name": "azure/gpt-5.4"}, "behavior_count": 5}
            first = prepare_artifact_plan(
                ctx=ctx,
                stage_name="policy",
                raw_cfg=raw_cfg,
                forced=False,
            )
            activate_artifact_plan(ctx, first)
            first.output_paths["policy"].parent.mkdir(parents=True, exist_ok=True)
            first.output_paths["policy"].write_text('{"behaviors":[]}', encoding="utf-8")
            first.output_paths["systematization"].write_text("{}", encoding="utf-8")
            finalize_artifact_plan(ctx, first)

            changed_ctx = self._ctx(root)
            changed_ctx["concept"] = "Changed concept text."
            second = prepare_artifact_plan(
                ctx=changed_ctx,
                stage_name="policy",
                raw_cfg=raw_cfg,
                forced=False,
            )

            self.assertFalse(second.reused)
            self.assertEqual(second.version, "v0002")


if __name__ == "__main__":
    unittest.main()
