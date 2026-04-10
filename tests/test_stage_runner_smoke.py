import unittest
from pathlib import Path

from p2m.core.config_model import EvaluationConfig, JudgeConfig, TargetConfig
from p2m.stages import judge, policy, rollout, seeds, systematization, systematization_convert
from tests.helpers import StageSmokeCase, run_stage_smoke_case, write_json, write_jsonl


def _common_context(root: Path) -> dict[str, object]:
    return {
        "suite_id": "suite-1",
        "run_id": "run-1",
        "risk": "Harmful advice",
        "risk_name": "harmful_advice",
        "suite_root": root,
        "run_root": root / "run-1",
        "artifacts_root": root,
        "config_path": root / "config.yaml",
        "strict": False,
    }


def _policy_case() -> StageSmokeCase:
    def cfg_factory(root: Path) -> dict[str, object]:
        return {
            "model": {
                "name": "azure/gpt-5.4",
                "temperature": 0.0,
                "max_tokens": 800,
            },
            "sub_risk_count": 5,
            "save_dir": str(root),
        }

    def context_factory(root: Path) -> dict[str, object]:
        return _common_context(root)

    def result_factory(root: Path, _kwargs: dict[str, object]) -> dict[str, str]:
        out_path = root / "policy.json"
        out_path.write_text("{}", encoding="utf-8")
        return {"policy_path": str(out_path)}

    def assert_fn(calls: dict[str, object], result: object, root: Path) -> None:
        assert calls["risk"] == "Harmful advice"
        assert calls["model"] == "azure/gpt-5.4"
        assert result["policy_path"] == str(root / "policy.json")

    return StageSmokeCase(
        name="policy",
        run=policy.run,
        workflow_patch="p2m.stages.policy.run_policy",
        cfg_factory=cfg_factory,
        context_factory=context_factory,
        result_factory=result_factory,
        assert_fn=assert_fn,
    )


def _seeds_case() -> StageSmokeCase:
    def cfg_factory(root: Path) -> dict[str, object]:
        return {
            "prompt": {"model": {"name": "azure/gpt-5.4"}, "budget": 10},
            "policy_path": str(root / "policy.json"),
            "save_path": str(root / "seeds.jsonl"),
        }

    def context_factory(root: Path) -> dict[str, object]:
        return _common_context(root)

    def result_factory(root: Path, _kwargs: dict[str, object]) -> dict[str, str]:
        out_path = root / "seeds.jsonl"
        out_path.write_text("", encoding="utf-8")
        return {"seeds_path": str(out_path)}

    def assert_fn(calls: dict[str, object], result: object, root: Path) -> None:
        assert calls["prompt"]["model"] == "azure/gpt-5.4"
        assert calls["prompt"]["budget"] == 10
        assert result["seeds_path"] == str(root / "seeds.jsonl")

    return StageSmokeCase(
        name="seeds",
        run=seeds.run,
        workflow_patch="p2m.stages.seeds.run_seeds",
        cfg_factory=cfg_factory,
        context_factory=context_factory,
        result_factory=result_factory,
        assert_fn=assert_fn,
    )


def _rollout_case() -> StageSmokeCase:
    def setup_fn(root: Path) -> None:
        write_jsonl(root / "seeds.jsonl", [{"kind": "prompt", "seed": {"description": "seed prompt"}}])

    def cfg_factory(root: Path) -> dict[str, object]:
        return {
            "seed_path": str(root / "seeds.jsonl"),
            "save_dir": str(root),
            "max_tokens": 1000,
            "strict": False,
        }

    def context_factory(root: Path) -> dict[str, object]:
        context = _common_context(root)
        context["target"] = TargetConfig(model="azure/gpt-5.4")
        context["evaluation"] = EvaluationConfig(judge=JudgeConfig(model="azure/gpt-5.4"))
        return context

    def result_factory(root: Path, _kwargs: dict[str, object]) -> dict[str, str]:
        transcripts = root / "transcripts.jsonl"
        transcripts.write_text("", encoding="utf-8")
        return {"transcripts_path": str(transcripts)}

    def assert_fn(calls: dict[str, object], result: object, root: Path) -> None:
        assert calls["target"].model == "azure/gpt-5.4"
        assert result["transcripts_path"] == str(root / "transcripts.jsonl")

    return StageSmokeCase(
        name="rollout",
        run=rollout.run,
        workflow_patch="p2m.stages.rollout.run_rollout",
        cfg_factory=cfg_factory,
        context_factory=context_factory,
        result_factory=result_factory,
        assert_fn=assert_fn,
        setup_fn=setup_fn,
    )


def _judge_case() -> StageSmokeCase:
    def setup_fn(root: Path) -> None:
        write_json(root / "policy.json", {"risk": {"name": "Risk"}, "sub_risks": []})
        write_jsonl(root / "transcripts.jsonl", [{"kind": "prompt", "seed_id": "seed-1"}])

    def cfg_factory(root: Path) -> dict[str, object]:
        return {
            "transcripts_path": str(root / "transcripts.jsonl"),
            "policy_path": str(root / "policy.json"),
            "save_dir": str(root),
            "strict": False,
        }

    def context_factory(root: Path) -> dict[str, object]:
        context = _common_context(root)
        context["target"] = TargetConfig(model="azure/gpt-5.4")
        context["evaluation"] = EvaluationConfig(judge=JudgeConfig(model="azure/gpt-5.4"))
        return context

    def result_factory(root: Path, _kwargs: dict[str, object]) -> dict[str, str]:
        scores = root / "scores.jsonl"
        metrics = root / "metrics.json"
        scores.write_text("", encoding="utf-8")
        metrics.write_text("{}", encoding="utf-8")
        return {"scores_path": str(scores), "metrics_path": str(metrics)}

    def assert_fn(calls: dict[str, object], result: object, root: Path) -> None:
        assert calls["evaluation"].judge.model == "azure/gpt-5.4"
        assert result["scores_path"] == str(root / "scores.jsonl")
        assert result["metrics_path"] == str(root / "metrics.json")

    return StageSmokeCase(
        name="judge",
        run=judge.run,
        workflow_patch="p2m.stages.judge.run_judge",
        cfg_factory=cfg_factory,
        context_factory=context_factory,
        result_factory=result_factory,
        assert_fn=assert_fn,
        setup_fn=setup_fn,
    )


def _systematization_case() -> StageSmokeCase:
    def cfg_factory(root: Path) -> dict[str, object]:
        return {
            "concept": "harmful advice",
            "model": "azure/gpt-5.4",
            "mode": "research",
            "reasoning_effort": "high",
            "temperature": 0.0,
            "max_tokens": 1600,
            "save_path": str(root / "systematization.json"),
        }

    def context_factory(root: Path) -> dict[str, object]:
        return _common_context(root)

    def result_factory(root: Path, kwargs: dict[str, object]) -> Path:
        out_path = Path(kwargs["save_path"])
        out_path.write_text("{}", encoding="utf-8")
        return out_path

    def assert_fn(calls: dict[str, object], result: object, root: Path) -> None:
        assert calls["concept"] == "harmful advice"
        assert calls["mode"] == "research"
        assert calls["reasoning_effort"] == "high"
        assert result["systematization_path"] == str((root / "systematization.json").resolve())

    return StageSmokeCase(
        name="systematization",
        run=systematization.run,
        workflow_patch="p2m.stages.systematization.run_systematization",
        cfg_factory=cfg_factory,
        context_factory=context_factory,
        result_factory=result_factory,
        assert_fn=assert_fn,
    )


def _systematization_convert_case() -> StageSmokeCase:
    def cfg_factory(root: Path) -> dict[str, object]:
        return {
            "systematization_path": str(root / "systematization.json"),
            "save_path": str(root / "policy-copy.json"),
            "model": "azure/gpt-5.4",
            "max_tokens": 1200,
            "temperature": 0.0,
            "sub_risk_count_hint": 18,
        }

    def context_factory(root: Path) -> dict[str, object]:
        return _common_context(root)

    def result_factory(root: Path, kwargs: dict[str, object]) -> Path:
        out_path = Path(kwargs["save_path"])
        out_path.write_text("{}", encoding="utf-8")
        return out_path

    def assert_fn(calls: dict[str, object], result: object, root: Path) -> None:
        assert calls["model"] == "azure/gpt-5.4"
        assert calls["sub_risk_count_hint"] == 18
        assert result["policy_generated_path"] == str((root / "policy-copy.json").resolve())
        assert Path(result["policy_generated_path"]).exists()

    return StageSmokeCase(
        name="systematization_convert",
        run=systematization_convert.run,
        workflow_patch="p2m.stages.systematization_convert.run_systematization_to_policy",
        cfg_factory=cfg_factory,
        context_factory=context_factory,
        result_factory=result_factory,
        assert_fn=assert_fn,
    )


class StageRunnerSmokeTest(unittest.TestCase):
    def test_stage_runners_delegate_to_stage_functions(self) -> None:
        for case in [
            _policy_case(),
            _seeds_case(),
            _rollout_case(),
            _judge_case(),
            _systematization_case(),
            _systematization_convert_case(),
        ]:
            with self.subTest(stage=case.name):
                run_stage_smoke_case(case)


if __name__ == "__main__":
    unittest.main()
