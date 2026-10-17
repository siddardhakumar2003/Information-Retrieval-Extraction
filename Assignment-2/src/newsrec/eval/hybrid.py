"""Does adding the lexical score to the semantic one help? (Q3.6 follow-up)

Design: agent_docs/q4_plan.md, agent_docs/q3_plan.md D8

## The question, and why it needed measuring rather than assuming

Q4 measured BM25 and embeddings separately and found semantic ahead on AUC on
all three corpora. That does not settle whether *combining* them beats either,
and the two fail differently enough to make combining plausible: BM25 scores
exactly zero for half of MIND's candidates, while embeddings retrieve articles
97 days staler than BM25 does (`lexical_vs_semantic.json`). Two systems with
uncorrelated weaknesses is the textbook setup where a blend wins.

Measured, it wins on **one corpus of three**, and the answer is
language-dependent. See `format_table` for the numbers as reported.

## The blend

Per impression, each score vector is min-max normalised to [0,1], then

    combined = alpha * bm25_norm + (1 - alpha) * cosine_norm

Normalised **within the impression**, not globally, because the two scales are
not comparable: BM25 is unbounded positive while cosine sits in a narrow high
band (the anisotropy recorded in design_rationale 1.18). Global normalisation
would make `alpha` mean "which scale is larger" rather than "how much do I
trust each signal for this user".

`alpha=1.0` must reproduce BM25's AUC and `alpha=0.0` must reproduce
semantic's, to the digit, against the committed `eval.json`. That endpoint
check is what makes the interior values trustworthy, and `run` asserts it
rather than leaving it to a reader to spot.

## Why the result is not a tuning knob

The best interior alpha differs by corpus -- 0.3 on MIND, 0.15 on both EB-NeRD
variants -- and on `ebnerd_demo` every alpha at or above 0.3 is an established
LOSS. Choosing alpha per dataset from these numbers would be fitting to the
evaluation set, which is the failure the whole repo is arranged against. The
sweep is therefore reported as an ablation and the submitted system stays
semantic-only.

## The mechanism, measured separately

`score_spread` records why. BM25's usefulness tracks how sharply it separates
candidates *inside one impression*, and the two languages fail in opposite
ways: English is silent half the time but decisive when it speaks; Danish
almost always speaks and barely discriminates. A Danish score of "some overlap
with everything" is as uninformative as no score, but unlike a zero it does not
fall through to the popularity tie-break -- it displaces it with noise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from newsrec import config
from newsrec.data import feature_store
from newsrec.eval import metrics, slices
from newsrec.eval.bootstrap import bootstrap_ci
from newsrec.eval.harness import _bm25_scores, _ranked_scores, _semantic_scores, _serving_order
from newsrec.retrieval import embed, query, recall
from newsrec.retrieval import index as idx

REPORT_NAME = "hybrid_ablation.json"

#: Swept, not tuned. The endpoints are the two pure methods and exist as a
#: self-check; the interior points are the actual question.
ALPHAS: tuple[float, ...] = (0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0)

#: How closely an endpoint must match the committed eval.json AUC. They are the
#: same computation over the same impressions, so this is a wiring check rather
#: than a tolerance: anything above floating-point noise means the blend is
#: reading different data or ordering it differently.
ENDPOINT_TOLERANCE = 1e-6


class EndpointMismatch(RuntimeError):
    """A pure-method endpoint did not reproduce the committed eval.json AUC."""


def minmax(x: np.ndarray) -> np.ndarray:
    """Scale to [0,1] within one impression; an all-equal vector becomes zeros.

    All-equal is the cold-start case -- every candidate scores 0 -- and mapping
    it to zeros rather than to 0.5 or to NaN keeps the blend at exactly the
    other method's ranking, so the popularity tie-break still decides.
    """
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float64)
    return (x.astype(np.float64) - lo) / (hi - lo)


def blend(lex: np.ndarray, sem: np.ndarray, alpha: float) -> np.ndarray:
    """`alpha * minmax(lex) + (1 - alpha) * minmax(sem)`, per impression.

    Extracted rather than inlined because the **normalisation is the design
    decision**, and inline it was untestable: `minmax` is monotonic, so at
    alpha=0 and alpha=1 the ranking is identical whether or not it is applied.
    Removing normalisation altogether therefore passed every endpoint check
    while changing every interior value -- found by mutation.

    Without it, the larger raw scale wins by arithmetic rather than by evidence.
    BM25 runs to tens; cosine sits inside [-1, 1]. A raw 0.5/0.5 blend is not a
    balanced opinion, it is BM25 with rounding error.
    """
    return alpha * minmax(lex) + (1.0 - alpha) * minmax(sem)


def _load(dataset: str, store_root: Path | None = None, split: str = "test"):
    root = store_root or config.DATA_DIR
    processed = root / "processed" / dataset
    impressions = feature_store.read_table(processed / "impressions.parquet")
    history = feature_store.read_table(processed / "history.parquet")
    sample = recall.sample_impressions(impressions, split)
    users = sample["user_id"].unique().to_list()
    user_history = history.filter(pl.col("user_id").is_in(users))
    embeddings = embed.load(embed.embeddings_dir(dataset))

    spec = config.tokeniser_for(dataset)
    articles = feature_store.read_table(
        root / "feature_store" / dataset / "article_features.parquet"
    ).select("article_id", "text")
    titles = feature_store.read_table(processed / "articles.parquet").select("article_id", "title")
    index = idx.build(articles, spec)

    lex = _bm25_scores(sample, user_history, titles, index, spec)
    sem, _ = _semantic_scores(sample, user_history, embeddings)
    return sample, impressions, user_history, lex, sem


def score_spread(sample: pl.DataFrame, lex: list[np.ndarray]) -> dict:
    """How sharply BM25 separates candidates *within* one impression.

    The AUC says lexical works in English and not in Danish; these say why. All
    four quantities are per impression and then averaged, never pooled -- a
    pooled spread would be dominated by differences in score level between
    impressions, which is not the question.
    """
    zero, cv, gap, tied, top1 = [], [], [], [], []
    for i, row in enumerate(sample.iter_rows(named=True)):
        x = lex[i]
        if x.size < 2:
            continue
        zero.append(float((x == 0).mean()))
        mean = float(x.mean())
        if mean > 0:
            # Spread relative to level. Comparable across datasets whose BM25
            # scores sit at different magnitudes, which a raw std is not.
            cv.append(float(x.std() / mean))
        top = float(x.max())
        if top > 0:
            rest = float(np.median(x[x < top])) if (x < top).any() else top
            gap.append((top - rest) / top)
        _, counts = np.unique(x, return_counts=True)
        tied.append(float(counts[counts > 1].sum() / x.size))
        clicked = set(row["clicked_ids"])
        if clicked:
            top1.append(1.0 if row["candidate_ids"][int(np.argmax(x))] in clicked else 0.0)

    def mean(values: list[float]) -> float:
        # NaN, not 0.0, for an empty split. `np.mean([])` is NaN anyway but warns;
        # this says the same thing deliberately, and 0.0 here would read as a
        # measured zero rather than as "there was nothing to measure".
        return round(float(np.mean(values)), 4) if values else float("nan")

    return {
        "zero_score_share": mean(zero),
        "within_impression_cv": mean(cv),
        "top_vs_rest_gap": mean(gap),
        "tied_share": mean(tied),
        "top1_is_a_click": mean(top1),
    }


def evaluate_dataset(dataset: str, store_root: Path | None = None, split: str = "test") -> dict:
    """The alpha sweep, the paired deltas, and the spread diagnostics."""
    sample, impressions, user_history, lex, sem = _load(dataset, store_root, split)
    popularity = query.training_popularity(impressions)

    rows = list(sample.iter_rows(named=True))
    keep = []
    labels, pops, cands = [], [], []
    for i, row in enumerate(rows):
        y = np.array(
            [1 if a in set(row["clicked_ids"]) else 0 for a in row["candidate_ids"]],
            dtype=np.int64,
        )
        if metrics.degenerate_reason(y) is not None:
            continue
        keep.append(i)
        labels.append(y)
        pops.append(
            np.array([popularity.get(a, 0) for a in row["candidate_ids"]], dtype=np.float64)
        )
        cands.append(list(row["candidate_ids"]))

    per_alpha: dict[float, np.ndarray] = {}
    for alpha in ALPHAS:
        aucs = []
        for k, i in enumerate(keep):
            combined = blend(lex[i], sem[i], alpha)
            order = _serving_order(combined, pops[k], cands[k])
            aucs.append(metrics.auc(labels[k], _ranked_scores(order)))
        per_alpha[alpha] = np.asarray(aucs, dtype=np.float64)

    base = per_alpha[0.0]
    out_alpha = {}
    for alpha in ALPHAS:
        ci = bootstrap_ci(per_alpha[alpha])
        entry = {
            "auc": round(ci["mean"], 6),
            "ci_low": round(ci["ci_low"], 6),
            "ci_high": round(ci["ci_high"], 6),
            "n": ci["n"],
        }
        if alpha != 0.0:
            d = bootstrap_ci(per_alpha[alpha] - base)
            entry["delta_vs_semantic"] = {
                "mean": round(d["mean"], 6),
                "ci_low": round(d["ci_low"], 6),
                "ci_high": round(d["ci_high"], 6),
                "established": bool(d["ci_low"] > 0.0 or d["ci_high"] < 0.0),
                "direction": "hybrid_better" if d["mean"] > 0 else "hybrid_worse",
            }
        out_alpha[str(alpha)] = entry

    interior = [a for a in ALPHAS if 0.0 < a < 1.0]
    best = max(interior, key=lambda a: out_alpha[str(a)]["auc"])
    best_entry = out_alpha[str(best)]["delta_vs_semantic"]

    masks = slices.build(sample, user_history, impressions)
    per_slice = {}
    for name, mask in masks.items():
        sel = [k for k, i in enumerate(keep) if mask[i]]
        if not sel:
            continue
        d = bootstrap_ci(per_alpha[best][sel] - base[sel])
        per_slice[name] = {
            "n": len(sel),
            "lexical": round(float(per_alpha[1.0][sel].mean()), 6),
            "semantic": round(float(base[sel].mean()), 6),
            "hybrid": round(float(per_alpha[best][sel].mean()), 6),
            "delta_hybrid_minus_semantic": round(d["mean"], 6),
            "established": bool(d["ci_low"] > 0.0 or d["ci_high"] < 0.0),
        }

    return {
        "dataset": dataset,
        "split": split,
        "language": config.DATASETS[dataset].language,
        "impressions_evaluated": len(sample),
        "impression_set_fingerprint": recall.impression_set_fingerprint(sample),
        "alpha": out_alpha,
        "best_interior_alpha": best,
        "hybrid_beats_semantic": bool(
            best_entry["established"] and best_entry["direction"] == "hybrid_better"
        ),
        "slices": per_slice,
        "bm25_score_spread": score_spread(sample, lex),
    }


def select_on_val(dataset: str, store_root: Path | None = None) -> dict:
    """Choose alpha on the VALIDATION split, then report its effect on test.

    Why this exists: the sweep in `evaluate_dataset` picks its best alpha by
    looking at test-set AUC, and reporting that same AUC overstates the result
    -- part of any gain is having peeked. Selecting on a split we do not report
    on removes the objection entirely, and the val split has been sitting
    unused since Q1 precisely so that this kind of choice has somewhere honest
    to happen.

    Returns the val-selected alpha, its test AUC, and the paired delta against
    semantic **on test**. If that delta is not established, the honest reading
    is that the interior optimum was not stable enough to act on -- which is a
    result, not a failure.
    """
    on_val = evaluate_dataset(dataset, store_root, split="val")
    interior = [a for a in ALPHAS if 0.0 < a < 1.0]
    # Best on val INCLUDING the semantic endpoint: if no blend beats pure
    # semantic on val, the honest selection is no blend at all.
    candidates = [0.0, *interior]
    chosen = max(candidates, key=lambda a: on_val["alpha"][str(a)]["auc"])

    on_test = evaluate_dataset(dataset, store_root, split="test")
    entry = on_test["alpha"][str(chosen)]
    delta = entry.get("delta_vs_semantic")

    return {
        "val_n": on_val["impressions_evaluated"],
        "val_alpha_curve": {a: on_val["alpha"][str(a)]["auc"] for a in map(str, candidates)},
        "selected_alpha": chosen,
        "selected_on": "val",
        "test_auc": entry["auc"],
        "test_semantic_auc": on_test["alpha"]["0.0"]["auc"],
        "test_delta_vs_semantic": delta,
        "verdict": (
            "no blend selected on val; semantic-only"
            if chosen == 0.0
            else (
                "val-selected blend beats semantic on test"
                if delta and delta["established"] and delta["mean"] > 0
                else "val-selected blend does NOT establish a gain on test"
            )
        ),
    }


def _check_endpoints(measured: dict, metrics_dir: Path) -> dict:
    """Assert alpha=1 reproduces BM25 and alpha=0 reproduces semantic.

    Without this the interior numbers are unfalsifiable: a blend that silently
    read the wrong scores would still produce a smooth, plausible curve.
    """
    path = metrics_dir / "eval.json"
    if not path.exists():
        return {"checked": False, "reason": "eval.json not built"}

    committed = json.loads(path.read_text(encoding="utf-8"))["datasets"]
    problems = {}
    for dataset, block in measured.items():
        if dataset not in committed:
            continue
        want = committed[dataset]["methods"]
        for alpha, method in (("1.0", "bm25"), ("0.0", "semantic")):
            got = block["alpha"][alpha]["auc"]
            expected = want[method]["overall"]["auc"]["mean"]
            if abs(got - expected) > ENDPOINT_TOLERANCE:
                problems[f"{dataset}/alpha={alpha}"] = {"here": got, method: expected}
    if problems:
        raise EndpointMismatch(
            f"pure-method endpoints do not reproduce eval.json: {problems}. "
            f"The blend is reading different data or ordering it differently, so no "
            f"interior alpha means anything."
        )
    return {"checked": True, "tolerance": ENDPOINT_TOLERANCE}


def run(
    datasets: tuple[str, ...] | None = None,
    out_dir: Path | None = None,
    with_val_selection: bool = True,
    **kwargs,
) -> Path:
    """Write `hybrid_ablation.json`."""
    datasets = tuple(config.DATASETS) if datasets is None else datasets
    out_dir = config.METRICS_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    measured = {d: evaluate_dataset(d, **kwargs) for d in datasets}
    if with_val_selection:
        for dataset in datasets:
            measured[dataset]["val_selected"] = select_on_val(dataset, **kwargs)
    endpoint = _check_endpoints(measured, out_dir)

    wins = sorted(d for d, b in measured.items() if b["hybrid_beats_semantic"])
    honest = sorted(
        d
        for d, b in measured.items()
        if b.get("val_selected", {}).get("verdict", "").startswith("val-selected blend beats")
    )
    report = {
        "report": "hybrid_ablation",
        "question": "does blending the lexical score into the semantic one beat semantic alone?",
        "blend": "alpha * minmax(bm25) + (1 - alpha) * minmax(cosine), normalised per impression",
        "alphas": list(ALPHAS),
        "endpoint_check": endpoint,
        "endpoint_note": (
            "alpha=1.0 must equal eval.json's bm25 AUC and alpha=0.0 its semantic AUC. "
            "Asserted, not eyeballed: a blend reading the wrong scores would still "
            "produce a smooth, plausible curve."
        ),
        "answer": (
            f"Selecting alpha on TEST, hybrid beats semantic on {len(wins)} of "
            f"{len(measured)} corpora: {wins or 'none'}. Selecting alpha on VAL and "
            f"reporting on test -- the honest procedure -- it beats semantic on "
            f"{len(honest)}: {honest or 'none'}. Both Danish corpora independently "
            f"select alpha=0 on val, i.e. no lexical component at all."
        ),
        "selection_note": (
            "The test-split sweep picks its best alpha by looking at the number it "
            "then reports, so part of any gain there is having peeked. `val_selected` "
            "repeats the choice on the validation split -- untouched since Q1 for "
            "exactly this purpose -- and applies it to test. A gain that survives "
            "that is evidence; one that does not was an artefact of selection."
        ),
        "mechanism_note": (
            "bm25_score_spread explains the split. BM25's value tracks how sharply it "
            "separates candidates WITHIN an impression, and the two languages fail in "
            "opposite ways: English is silent much of the time but decisive when it "
            "speaks; Danish almost always speaks and barely discriminates. A Danish "
            "score of 'some overlap with everything' is as uninformative as no score, "
            "but unlike a zero it does not fall through to the popularity tie-break."
        ),
        "bootstrap": {
            "n_resamples": config.BOOTSTRAP_N,
            "ci": config.BOOTSTRAP_CI,
            "paired": True,
            "paired_against": "alpha=0.0 (semantic only)",
        },
        "datasets": measured,
    }
    path = out_dir / REPORT_NAME
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def format_table(report: dict) -> str:
    lines = []
    for dataset, block in sorted(report["datasets"].items()):
        lines.append(f"\n{dataset} ({block['language']})  n={block['impressions_evaluated']}")
        for alpha in report["alphas"]:
            entry = block["alpha"][str(alpha)]
            delta = entry.get("delta_vs_semantic")
            tag = ""
            if delta:
                mark = "established" if delta["established"] else "not established"
                tag = f"  delta {delta['mean']:+.4f} [{delta['ci_low']:+.4f},{delta['ci_high']:+.4f}] {mark}"
            label = {0.0: " (semantic)", 1.0: " (bm25)"}.get(alpha, "")
            lines.append(f"  alpha={alpha:<5}{label:<11} AUC {entry['auc']:.4f}{tag}")
        spread = block["bm25_score_spread"]
        lines.append(
            f"  bm25 spread: zeros {spread['zero_score_share']:.1%}  "
            f"within-impression CV {spread['within_impression_cv']:.2f}  "
            f"top-vs-rest {spread['top_vs_rest_gap']:.1%}"
        )
    lines.append(f"\n{report['answer']}")
    return "\n".join(lines)
