"""Q2.9 — the tokenisation ablation.

Requirement: the language-aware tokenisation choice is *declared* in
`q2_plan.md` D4, and the stemmed variant is **scored as a logged ablation**
rather than silently swapped in. This module is the scoring.

## What is being compared, and why it is not obvious which should win

For MIND (English) the baseline already lowercases and removes stopwords; the
variant adds the Porter stemmer, which strips suffixes so that `running`,
`runs` and `run` collapse to one token.

For EB-NeRD (Danish) the baseline is deliberately naive — lowercase and split on
word characters, nothing else — and the variant adds a Danish stopword list plus
the Snowball Danish stemmer. Danish is morphologically richer than English:
it compounds freely and marks definiteness with a suffix, so `bil`, `bilen`
("the car") and `biler` ("cars") are three unrelated tokens to a naive
tokeniser. Collapsing them should help.

**Should** is the operative word, and it is why this is measured. Stemming is a
lossy merge: it also collapses words that genuinely differ, and in a corpus of
short news titles — where a query is a handful of terms — losing one distinction
can cost more than gaining three matches.

That this needed measuring is not hypothetical. During Q2's diagnosis of the low
recall numbers, an ad-hoc single-variant run reported the Danish stemmed variant
scoring *below* the naive baseline (0.0300 against 0.0333 at K=200). Run properly
here — same seeded sample for both arms, difference bootstrapped per impression —
the sign is the other way (+0.0070 at K=200 on `ebnerd_demo`) and the interval
still spans zero. The anecdote was not merely imprecise, it pointed the wrong
way. This module exists so the design note quotes the second kind of number.

## Paired, because two overlapping intervals answer nothing

Both variants are evaluated on **the same impressions** — the same seeded sample
`recall.sample_impressions` gives Q2.8, with the fingerprint recorded so it can
be checked rather than assumed. That makes the comparison paired, and the
difference is bootstrapped per impression.

The distinction matters. Suppose a variant improves nearly every impression by a
small amount. Each variant's own 95% interval spans the variation *between
impressions*, which is large — some users are easy, most are not — so the two
intervals overlap heavily and an unpaired reading says "no difference
established". The interval on the per-impression *difference* spans only the
variation in the effect itself, which can be small and clearly non-zero. Pairing
is free here because the impression set is identical by construction; not using
it would be discarding evidence.

`significant` is reported as "does the difference's 95% interval exclude zero",
and nothing else. It is not a p-value and the design note must not call it one.

## One definition of recall, reached two ways

Nothing here recomputes recall. `recall.evaluate_dataset` is called with a
`spec` override, so both variants go through the identical scoring, sampling,
pooling and exclusion rules. A second implementation would mean a difference
between variants and a difference between two measurement paths were
indistinguishable.
"""

from __future__ import annotations

import json
from pathlib import Path

from newsrec import config
from newsrec.eval.bootstrap import bootstrap_ci
from newsrec.retrieval import recall, tokenise

REPORT_NAME = "bm25_tokenisation_ablation.json"


def compare_variants(
    dataset: str,
    variants: tuple[config.TokeniserSpec, ...] | None = None,
    **kwargs,
) -> dict:
    """Score every tokenisation variant for one dataset and pair the results.

    `variants` defaults to `config.tokeniser_variants_for(dataset)`, headline
    first. The first entry is the reference every delta is measured against.
    """
    variants = config.tokeniser_variants_for(dataset) if variants is None else variants
    if not variants:
        raise ValueError(f"no tokenisation variants registered for {dataset!r}")

    measured = {v.name: recall.evaluate_dataset(dataset, spec=v, **kwargs) for v in variants}

    baseline_name = variants[0].name
    baseline = measured[baseline_name]

    # The whole point of the pairing is that this holds. Asserted rather than
    # trusted: if a future change made the variants sample differently, the
    # per-impression subtraction below would quietly compare unrelated rows and
    # produce a delta that looks measured but means nothing.
    fingerprints = {m["impression_set_fingerprint"] for m in measured.values()}
    if len(fingerprints) != 1:
        raise ValueError(
            f"{dataset}: variants were evaluated on different impression sets "
            f"({sorted(fingerprints)}); the paired delta would be meaningless"
        )

    deltas: dict[str, dict[str, dict]] = {}
    for name, m in measured.items():
        if name == baseline_name:
            continue
        per_k = {}
        for k, values in m["per_impression_recall"].items():
            base = baseline["per_impression_recall"][k]
            diff = [a - b for a, b in zip(values, base, strict=True)]
            ci = bootstrap_ci(diff)
            per_k[k] = {
                **ci,
                "k": int(k),
                # Both bounds on the same side of zero. Says the sign of the
                # effect is established at this sample size; says nothing about
                # whether the effect is large enough to care about.
                "significant": bool(ci["ci_low"] > 0.0 or ci["ci_high"] < 0.0),
            }
        deltas[name] = per_k

    for m in measured.values():
        m.pop("per_impression_recall")
        m.pop("timing_seconds")

    return {
        "dataset": dataset,
        "language": config.DATASETS[dataset].language,
        "baseline_variant": baseline_name,
        "impression_set_fingerprint": fingerprints.pop(),
        "variants": {
            name: {
                "spec": {
                    "name": v.name,
                    "language": v.language,
                    "lowercase": v.lowercase,
                    "stopwords": v.stopwords,
                    "stemmer": v.stemmer,
                },
                "tokeniser": tokenise.fingerprint(v),
                "recall": measured[name]["recall"],
                "corpus_articles": measured[name]["corpus_articles"],
                "cold_start_impressions": measured[name]["cold_start_impressions"],
            }
            for name, v in ((v.name, v) for v in variants)
        },
        "delta_vs_baseline": deltas,
    }


def run(
    datasets: tuple[str, ...] | None = None,
    out_dir: Path | None = None,
    **kwargs,
) -> Path:
    """Write `reports/metrics/bm25_tokenisation_ablation.json` (Q2.9).

    Runs for BOTH datasets, not only EB-NeRD. Q2.9's checkable condition names
    `ebnerd_*`, because the Danish baseline is the deliberately naive one — but
    the English stem variant costs one more pass and turns "stemming helps
    morphologically rich languages" from an assertion into a contrast with a
    control.

    Carries no wall-clock timings, so two runs of unchanged code produce
    identical bytes (Q1.13) — the rule `recall.run` learned the hard way.
    """
    datasets = tuple(config.DATASETS) if datasets is None else datasets
    out_dir = config.METRICS_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "ablation": "tokenisation",
        "requirement": "Q2.9",
        "metric": "recall_at_k",
        "retriever": "bm25",
        "k_values": list(config.RECALL_K_VALUES),
        "paired": True,
        "bootstrap": {"n_resamples": config.BOOTSTRAP_N, "ci": config.BOOTSTRAP_CI},
        "datasets": {d: compare_variants(d, **kwargs) for d in datasets},
    }
    dest = out_dir / REPORT_NAME
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest


def format_table(report: dict) -> str:
    """The variant x K table, for the design note and the terminal."""
    lines = [
        f"{'dataset':<14}{'variant':<14}{'K':>6}{'recall':>9}"
        f"{'delta':>10}{'95% CI on delta':>24}{'sig':>5}",
        "-" * 82,
    ]
    for name, d in report["datasets"].items():
        for variant in d["variants"]:
            for k in report["k_values"]:
                r = d["variants"][variant]["recall"][str(k)]
                delta = d["delta_vs_baseline"].get(variant, {}).get(str(k))
                if delta is None:
                    shown, ci, sig = "—", "(baseline)", ""
                else:
                    shown = f"{delta['mean']:+.4f}"
                    ci = f"[{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}]"
                    sig = "yes" if delta["significant"] else "no"
                lines.append(
                    f"{name:<14}{variant:<14}{k:>6}{r['mean']:>9.4f}{shown:>10}{ci:>24}{sig:>5}"
                )
    return "\n".join(lines)
