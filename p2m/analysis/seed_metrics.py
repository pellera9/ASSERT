from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None

# Optional local embedding backend
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

# ---------------------- Helpers ----------------------


@dataclass
class Config:
    policy_path: str
    seeds_path: str
    embed_model: str
    embed_backend: str  # "openai" or "hf"
    k_list: List[int]
    example_distance_thresh: float
    presence_coverage: bool
    out_json: str
    out_md: Optional[str]


NEAR_DUP_THRESHOLD = 0.95


def _missing_dependency(name: str) -> ModuleNotFoundError:
    exc = ModuleNotFoundError(f"No module named '{name}'")
    exc.name = name
    return exc


def _require_numpy() -> Any:
    if np is None:
        raise _missing_dependency("numpy")
    return np


def _require_openai() -> Any:
    if OpenAI is None:
        raise _missing_dependency("openai")
    return OpenAI


def load_policy(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    sub_risks = data.get("sub_risks") or []
    flattened = []
    for idx, sr in enumerate(sub_risks):
        flattened.append(
            {
                "id": idx,
                "name": sr.get("name", f"sub_risk_{idx}"),
                "definition": sr.get("definition", ""),
                "examples": sr.get("examples", []) or [],
            }
        )
    return {"raw": data, "sub_risks": flattened}


def load_seeds(path: Path) -> List[Dict[str, Any]]:
    seeds = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        seeds.append(row)
    return seeds


def _prompt_seed_text(row: Dict[str, Any]) -> str | None:
    if row.get("kind") != "prompt":
        return None
    row_seed = row.get("seed")
    if isinstance(row_seed, dict):
        text = str(row_seed.get("description") or "").strip()
        return text or None
    return None


def pairwise_sims(mat: np.ndarray) -> np.ndarray:
    np_mod = _require_numpy()
    if mat.shape[0] < 2:
        return np_mod.array([])
    norm = mat / np_mod.linalg.norm(mat, axis=1, keepdims=True)
    sim = norm @ norm.T
    # extract upper triangle without diagonal
    iu = np_mod.triu_indices_from(sim, k=1)
    return sim[iu]


def vendi_score(mat: np.ndarray) -> float:
    """Effective set size via entropy of similarity matrix eigenvalues."""
    np_mod = _require_numpy()
    n = mat.shape[0]
    if n == 0:
        return 0.0
    # similarity matrix using cosine
    norm = mat / np_mod.linalg.norm(mat, axis=1, keepdims=True)
    k = norm @ norm.T
    eigvals = np_mod.linalg.eigvalsh(k / n)
    eigvals = np_mod.clip(eigvals, 1e-12, None)
    ent = -np_mod.sum(eigvals * np_mod.log(eigvals))
    return float(np_mod.exp(ent))


def gini(arr: List[float]) -> float:
    np_mod = _require_numpy()
    if not arr:
        return 0.0
    x = np_mod.array(arr)
    if np_mod.allclose(x, 0):
        return 0.0
    x_sorted = np_mod.sort(x)
    n = len(x)
    idx = np_mod.arange(1, n + 1)
    return float((2 * np_mod.sum(idx * x_sorted) / (n * np_mod.sum(x_sorted))) - (n + 1) / n)


def entropy(arr: List[float]) -> float:
    np_mod = _require_numpy()
    if not arr:
        return 0.0
    x = np_mod.array(arr, dtype=float)
    total = x.sum()
    if total <= 0:
        return 0.0
    p = x / total
    p = p[p > 0]
    return float(-np_mod.sum(p * np_mod.log(p)))


def embed_texts_openai(client: OpenAI, model: str, texts: List[str]) -> np.ndarray:
    np_mod = _require_numpy()
    if not texts:
        return np_mod.zeros((0, 1))
    resp = client.embeddings.create(model=model, input=texts)
    vecs = [np_mod.array(e.embedding, dtype=float) for e in resp.data]
    return np_mod.vstack(vecs)


def embed_texts_hf(model: str, texts: List[str]) -> np.ndarray:
    np_mod = _require_numpy()
    if not texts:
        return np_mod.zeros((0, 1))
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence_transformers not available; install it or use --embed-backend openai"
        )
    st_model = SentenceTransformer(model)
    vecs = st_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    return vecs


def embed_texts(
    client: OpenAI, backend: str, model: str, texts: List[str]
) -> np.ndarray:
    if backend == "hf":
        return embed_texts_hf(model, texts)
    return embed_texts_openai(client, model, texts)


def coverage_at_k(subrisk_counts: Dict[str, int], k: int, total_subrisks: int) -> float:
    if total_subrisks == 0:
        return 0.0
    hits = sum(1 for c in subrisk_counts.values() if c >= k)
    return hits / total_subrisks


# ---------------------- Main logic ----------------------


def compute_metrics(cfg: Config):
    np_mod = _require_numpy()
    tax = load_policy(Path(cfg.policy_path))
    sub_risks = tax["sub_risks"]
    subrisk_names = [sr["name"] for sr in sub_risks]

    seeds = load_seeds(Path(cfg.seeds_path))

    client = _require_openai()() if cfg.embed_backend == "openai" else None

    # Embed policy examples for coverage calc
    example_texts = []
    for sr in sub_risks:
        for ex in sr.get("examples", []):
            example_texts.append(ex)
    example_vecs = embed_texts(
        client, cfg.embed_backend, cfg.embed_model, example_texts
    )

    # Collect seed info
    per_subrisk_vecs: Dict[str, List[np.ndarray]] = defaultdict(list)
    per_subrisk_valid: Dict[str, List[bool]] = defaultdict(list)
    per_subrisk_len: Dict[str, List[int]] = defaultdict(list)

    prompt_seeds = [entry for entry in seeds if _prompt_seed_text(entry) is not None]
    seed_prompts = [_prompt_seed_text(entry) or "" for entry in prompt_seeds]
    seed_vecs = embed_texts(client, cfg.embed_backend, cfg.embed_model, seed_prompts)

    def embed_dim() -> int:
        if seed_vecs.size:
            return seed_vecs.shape[1]
        if example_vecs.size:
            return example_vecs.shape[1]
        return 1

    for entry, vec in zip(prompt_seeds, seed_vecs):
        sr = entry.get("sub_risk") or ""
        if sr not in subrisk_names:
            continue  # skip seeds whose sub_risk not in policy
        per_subrisk_vecs[sr].append(vec)
        per_subrisk_valid[sr].append(True)
        per_subrisk_len[sr].append(len(_prompt_seed_text(entry) or ""))

    # Metrics per subrisk
    per_stats = {}
    for sr in subrisk_names:
        vecs = per_subrisk_vecs.get(sr, [])
        valids = per_subrisk_valid.get(sr, [])
        lens = per_subrisk_len.get(sr, [])
        valid_vecs = [v for v, ok in zip(vecs, valids) if ok]
        valid_mat = np_mod.vstack(valid_vecs) if valid_vecs else np_mod.zeros((0, embed_dim()))

        # coverage count
        valid_count = len(valid_vecs)
        total = len(vecs)
        valid_rate = valid_count / total if total else 0.0

        # near-dup rate
        sims = pairwise_sims(valid_mat)
        near_dup_rate = float(np_mod.mean(sims > NEAR_DUP_THRESHOLD)) if sims.size else 0.0

        # vendi
        vendi = vendi_score(valid_mat) if valid_count > 0 else 0.0
        vendi_norm = vendi / valid_count if valid_count > 0 else 0.0

        # length stats
        mean_len = statistics.fmean(lens) if lens else 0.0
        var_len = statistics.pvariance(lens) if len(lens) > 1 else 0.0

        per_stats[sr] = {
            "valid_count": valid_count,
            "total_count": total,
            "valid_rate": valid_rate,
            "near_dup_rate": near_dup_rate,
            "vendi": vendi,
            "vendi_norm": vendi_norm,
            "mean_len": mean_len,
            "var_len": var_len,
        }

    # Overall coverage
    counts = {sr: per_stats[sr]["valid_count"] for sr in subrisk_names}
    if cfg.presence_coverage:
        covered = sum(1 for c in counts.values() if c > 0)
        coverage = {"presence": covered / len(subrisk_names) if subrisk_names else 0.0}
    else:
        coverage = {k: coverage_at_k(counts, k, len(subrisk_names)) for k in cfg.k_list}

    # Skew
    count_vals = list(counts.values())
    skew_gini = gini(count_vals)
    skew_ent = entropy(count_vals)

    # Example coverage
    ex_cov = {"fraction_within": 0.0, "median_distance": None}
    if example_vecs.shape[0] > 0:
        valid_all_list = [
            vec
            for sr in subrisk_names
            for vec, ok in zip(
                per_subrisk_vecs.get(sr, []), per_subrisk_valid.get(sr, [])
            )
            if ok
        ]
        if valid_all_list:
            valid_all = np_mod.vstack(valid_all_list)
            # compute min distance per example
            norm_valid = valid_all / np_mod.linalg.norm(valid_all, axis=1, keepdims=True)
            norm_examples = example_vecs / np_mod.linalg.norm(
                example_vecs, axis=1, keepdims=True
            )
            sims = norm_examples @ norm_valid.T
            max_sims = sims.max(axis=1)
            within = (1 - max_sims) <= cfg.example_distance_thresh  # cosine distance
            ex_cov["fraction_within"] = float(np_mod.mean(within))
            ex_cov["median_distance"] = float(np_mod.median(1 - max_sims))
        else:
            ex_cov["fraction_within"] = 0.0
            ex_cov["median_distance"] = None

    overall = {
        "coverage": coverage,
        "skew": {"gini": skew_gini, "entropy": skew_ent},
        "example_coverage": ex_cov,
        "valid_total": sum(count_vals),
        "seed_total": len(seeds),
    }

    report = {
        "config": asdict(cfg),
        "overall": overall,
        "per_sub_risk": per_stats,
    }

    Path(cfg.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out_json).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if cfg.out_md:
        lines = ["# Seed Metrics", "", "## Overall", ""]
        if cfg.presence_coverage:
            lines.append(f"Coverage: presence={coverage['presence']:.3f}")
        else:
            lines.append(
                "Coverage: "
                + ", ".join(f"@{k}={coverage[k]:.3f}" for k in cfg.k_list)
            )
        lines.append(f"Valid total: {overall['valid_total']} / {overall['seed_total']}")
        lines.append(f"Gini: {skew_gini:.3f}, Entropy: {skew_ent:.3f}")
        if ex_cov["median_distance"] is not None:
            lines.append(
                f"Example coverage: {ex_cov['fraction_within']:.3f}, median dist={ex_cov['median_distance']:.3f}"
            )
        else:
            lines.append("Example coverage: n/a")
        lines.append("")
        lines.append("## Per sub-risk")
        lines.append("")
        lines.append(
            "| sub_risk | valid | total | valid_rate | vendi | vendi/n | near_dup | mean_len | var_len |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for sr in subrisk_names:
            ps = per_stats[sr]
            lines.append(
                f"| {sr} | {ps['valid_count']} | {ps['total_count']} | {ps['valid_rate']:.3f} | "
                f"{ps['vendi']:.3f} | {ps['vendi_norm']:.3f} | {ps['near_dup_rate']:.3f} | "
                f"{ps['mean_len']:.1f} | {ps['var_len']:.1f} |"
            )
        Path(cfg.out_md).write_text("\n".join(lines), encoding="utf-8")
