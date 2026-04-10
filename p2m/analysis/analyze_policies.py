"""Analyze legacy policy-eval logs and produce summary stats and plots."""

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_log(path: Path) -> List[Dict[str, Any]]:
    """Extract per-sample records from a single .eval log."""
    rows: List[Dict[str, Any]] = []
    with zipfile.ZipFile(path) as zf:
        start = json.loads(zf.read("_journal/start.json"))
        eval_meta = start.get("eval", {})
        created = eval_meta.get("created")
        model = eval_meta.get("model")
        task_args = eval_meta.get("task_args", {})
        risk = task_args.get("risk", "unknown")
        validator_model = None
        scorers = eval_meta.get("scorers", [])
        if scorers:
            validator_model = scorers[0].get("options", {}).get("model_name")

        sample_names = [n for n in zf.namelist() if n.startswith("samples/")]
        for name in sample_names:
            sample = json.loads(zf.read(name))
            scores_block = sample.get("scores", {})
            if not scores_block:
                continue

            # Take the first judge entry
            scorer_key, scorer_entry = next(iter(scores_block.items()))
            validator = scorer_entry.get("metadata", {}).get("validator", {})
            dims = validator.get("scores", {}) or {}
            row = {
                "log": path.name,
                "created": created,
                "model": model,
                "validator_model": validator_model,
                "risk": risk,
                "epoch": sample.get("epoch"),
                "sample_id": sample.get("id"),
                "judge": scorer_key,
                "score_norm": scorer_entry.get("value"),
                "verdict": validator.get("verdict"),
                "overall_raw": validator.get("overall"),
                "overall_rationale": validator.get("overall_rationale"),
                "issues_count": len(validator.get("issues", []) or []),
            }
            # per-dimension scores
            row.update(
                {
                    "clarity": dims.get("clarity"),
                    "operationalizability": dims.get("operationalizability"),
                    "completeness": dims.get("completeness"),
                    "granularity": dims.get("granularity"),
                    "user_alignment": dims.get("user_alignment"),
                }
            )
            rows.append(row)
    return rows


def load_all(logs_dir: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for path in sorted(logs_dir.glob("*.eval")):
        try:
            records.extend(load_log(path))
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to load {path}: {exc}")
    if not records:
        raise SystemExit("No records loaded from logs")
    return pd.DataFrame(records)


def write_csv(df: pd.DataFrame, out_path: Path) -> None:
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


def plot_overall(df: pd.DataFrame, out_dir: Path) -> None:
    g = sns.catplot(
        data=df,
        x="model",
        y="overall_raw",
        kind="box",
        col="risk",
        col_wrap=2,
        sharey=True,
        height=4,
        aspect=1.1,
    )
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Overall (0-10) by model, faceted by risk")
    for ax in g.axes.flat:
        ax.set_ylabel("Overall score (0-10)")
        ax.tick_params(axis="x", rotation=15)
    out_path = out_dir / "overall_by_model_risk.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_dimensions(df: pd.DataFrame, out_dir: Path) -> None:
    dims = ["clarity", "operationalizability", "completeness", "granularity", "user_alignment"]
    melt = df.melt(
        id_vars=["model", "risk"],
        value_vars=dims,
        var_name="dimension",
        value_name="score",
    )
    g = sns.catplot(
        data=melt,
        x="dimension",
        y="score",
        hue="model",
        kind="bar",
        ci="sd",
        col="risk",
        col_wrap=2,
        sharey=True,
        height=4,
        aspect=1.3,
    )
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Mean dimension scores by model, faceted by risk")
    for ax in g.axes.flat:
        ax.set_ylabel("Score (0-10)")
        ax.tick_params(axis="x", rotation=20)
    out_path = out_dir / "dimensions_by_model.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")
