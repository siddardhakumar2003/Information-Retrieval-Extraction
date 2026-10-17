"""Thin argparse entrypoints that the Makefile calls.

This module holds *no* logic beyond argument parsing and dispatch. Every stage
body lives in the package, so `make bm25` and `python -m newsrec.cli bm25` run
byte-identical code paths.

Unimplemented stages raise NotImplementedError rather than returning quietly.
That is deliberate: a stage that no-ops writes nothing, the next stage reads
nothing, and the pipeline reports a plausible-looking zero instead of a failure.
`make all` must go red at the first missing stage.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from newsrec import config

#: stage -> (help text, requirement ids, design doc). The Makefile's targets
#: mirror these names one-to-one; if they drift, `make -n all` stops describing
#: what actually runs. The doc is named explicitly rather than derived from the
#: requirement id, because `ablations` is Q9 but is designed in eval_protocol.md.
STAGES: dict[str, tuple[str, str, str]] = {
    "data": ("parse, unify, temporally split, and build the feature store", "Q1", "q1_plan.md"),
    "bm25": ("build the inverted index, retrieve top-K, report recall@K", "Q2", "q2_plan.md"),
    # `embed` is its own stage rather than a phase inside `semantic` because it
    # produces its own artefact with its own cost. The Makefile already
    # distinguishes artifacts/embeddings/ from artifacts/ann/; running one
    # command for both stamps meant make could not tell which stage produced
    # what, and re-encoding (~31 min) would be triggered by an ANN parameter
    # change that does not touch a single vector.
    "embed": ("encode articles to unit-norm vectors", "Q3.1", "q3_plan.md"),
    "ann": ("build the HNSW index and measure it against exact search", "Q3.3", "q3_plan.md"),
    "semantic": ("user vectors, semantic recall@K", "Q3", "q3_plan.md"),
    "compare": ("paired lexical-vs-semantic, per dataset and per slice", "Q3.6", "q3_plan.md"),
    "eval": ("metrics, beyond-accuracy, slices, bootstrap CIs", "Q4", "q4_plan.md"),
    "hybrid": ("alpha sweep blending lexical into semantic", "Q3.6", "q4_plan.md"),
    "ablations": (
        "paired metrics with serving-unavailable features removed",
        "Q9.1",
        "eval_protocol.md",
    ),
    "submit": ("write prediction files for both Codabench competitions", "Q5", "q5_plan.md"),
    "behaviour_features": (
        "summarise click-history, session, and article behavioural features",
        "A2.Q1",
        "a2_q1_plan.md",
    ),
    "rerank": (
        "train GBDT/MLP re-rankers, ablate against the retrieval-only baseline",
        "A2.Q2",
        "a2_q2_plan.md",
    ),
    "nrms": (
        "reproduce NRMS, then one principled change, ablated with paired CIs",
        "A2.Q3",
        "a2_q3_plan.md",
    ),
    "rerank_eval": (
        "full metric set, slices and CIs for the two-stage pipeline",
        "A2.Q5",
        "a2_q5_plan.md",
    ),
    "serving": (
        "index memory, p99 single-request latency, cost/QPS, the 10x argument",
        "A2.Q4",
        "a2_q4_plan.md",
    ),
    "two_stage_ablation": (
        "re-ranker metrics with and without the serving-unavailable features",
        "A2.Q9.1",
        "a2_q5_plan.md",
    ),
}


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--dataset",
        choices=sorted(config.DATASETS),
        action="append",
        dest="datasets",
        help=(
            "dataset key; repeatable. Defaults to every registered dataset, "
            "because every question requires BOTH MIND and EB-NeRD."
        ),
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="auto detects CUDA and falls back to CPU, so `make all` completes without a GPU",
    )
    p.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    p.add_argument(
        "--force",
        action="store_true",
        help="rebuild even when the output is newer than its inputs",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="newsrec",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="stage", required=True, metavar="STAGE")
    for name, (help_text, reqs, _doc) in STAGES.items():
        p = sub.add_parser(name, help=f"[{reqs}] {help_text}")
        _add_common(p)
        if name == "embed":
            p.add_argument(
                "--diagnostics",
                action="store_true",
                help="measure the embedding space's geometry instead of encoding",
            )
        if name == "ann":
            p.add_argument(
                "--measure",
                action="store_true",
                help="[Q3.3] score the persisted indexes against exact search and write the report",
            )
        if name == "submit":
            # Only on this stage. The baseline exists to prove the file format
            # is accepted (OQ-6); it is not an alternative way to run anything
            # else, so no other subcommand offers it.
            p.add_argument(
                "--baseline",
                action="store_true",
                help="[OQ-6] write a trivial article_id-ordered file to prove the format",
            )
            p.add_argument(
                "--hybrid",
                action="store_true",
                help="[Q5.1] MIND only: blend lexical into semantic at the val-selected alpha",
            )
            p.add_argument(
                "--semantic",
                action="store_true",
                help="[Q5.2] EB-NeRD only: cosine ranking, no lexical term (val selected alpha=0)",
            )
        if name == "bm25":
            # Only on this stage, because it is Q2.9 specifically -- a flag on
            # every subcommand would imply the other stages honour it.
            p.add_argument(
                "--ablation",
                action="store_true",
                help="[Q2.9] score every tokenisation variant instead of the headline one",
            )
    return parser


def _run_data(datasets: list[str]) -> int:
    """Q1 — raw files to feature store, one dataset at a time."""
    from newsrec.data import feature_store

    for dataset in datasets:
        manifest = feature_store.build_dataset(dataset)
        counts, splits = manifest["counts"], manifest["splits"]
        print(
            f"{dataset}: {counts['parsed']:,} impressions "
            f"(train {splits['train']:,} / val {splits['val']:,} / test {splits['test']:,})"
            f"  rejected {counts['rejected']:,}  fingerprint {manifest['config_fingerprint']}"
        )
        if manifest["rejections"]:
            print(f"  rejections: {manifest['rejections']}")
    return 0


def _run_ann(datasets: list[str], measure: bool) -> int:
    """Q3.3 — build the ANN index, or measure it against exact search.

    Two modes because they are two artefacts: the per-dataset graph, and the
    single report covering all datasets. `--measure` reuses the persisted
    graphs rather than rebuilding them.
    """
    import json

    from newsrec.retrieval import ann

    if not measure:
        for dataset in datasets:
            dest = ann.build_and_save(dataset)
            print(f"{dataset}: ANN index -> {config.repo_relative(dest)}")
        return 0

    out = ann.run(datasets=tuple(datasets))
    report = json.loads(out.read_text())
    print(ann.format_table(report))
    print(f"\nwrote {config.repo_relative(out)}")
    return 0


def _run_compare(datasets: list[str]) -> int:
    """Q3.6 — paired lexical-vs-semantic, per dataset and per slice."""
    import json

    from newsrec.retrieval import compare

    out = compare.run(datasets=tuple(datasets))
    print(compare.format_table(json.loads(out.read_text())))
    print(f"\nwrote {config.repo_relative(out)}")
    return 0


def _run_submit_baseline() -> int:
    """OQ-6 — the trivial baseline, to prove the format before the real run."""
    from pathlib import Path

    from newsrec.submit import baseline

    summary = baseline.write_mind_baseline(config.PREDICTIONS_DIR)
    print(
        f"wrote {config.repo_relative(Path(summary['path']))}  "
        f"{summary['lines']:,} lines  {summary['rank_integers']:,} ranks  "
        f"{summary['bytes'] / 1e6:.1f} MB"
    )
    print("validated: line count, id order, and 1..n ranks on every line (Q5.3)")
    return 0


def _run_hybrid(datasets: list[str]) -> int:
    """Q3.6 follow-up — does blending lexical into semantic beat semantic alone?"""
    import json

    from newsrec.eval import hybrid

    out = hybrid.run(datasets=tuple(datasets))
    print(hybrid.format_table(json.loads(out.read_text())))
    print(f"\nwrote {config.repo_relative(out)}")
    return 0


def _run_submit_hybrid() -> int:
    """Q5.1 — the MIND hybrid submission at the val-selected blend weight."""
    from pathlib import Path

    from newsrec.submit import hybrid_predict

    summary = hybrid_predict.write_mind_hybrid(config.PREDICTIONS_DIR)
    print(
        f"wrote {config.repo_relative(Path(summary['path']))}  "
        f"{summary['lines']:,} lines  alpha={summary['alpha']}  "
        f"cold rows {summary['coverage']['both_silent']:,}"
    )
    print(f"archive: {summary['archive']['archive']} -> {summary['archive']['member']}")
    return 0


def _run_submit_semantic() -> int:
    """Q5.2 — the EB-NeRD semantic submission."""
    from pathlib import Path

    from newsrec.submit import semantic_predict

    summary = semantic_predict.write_ebnerd_semantic(config.PREDICTIONS_DIR)
    print(
        f"wrote {config.repo_relative(Path(summary['path']))}  "
        f"{summary['lines']:,} lines  "
        f"coverage candidates {summary['candidate_coverage']:.4f} "
        f"users {summary['user_coverage']:.4f}"
    )
    print(f"archive: {summary['archive']['archive']} -> {summary['archive']['member']}")
    return 0


def _run_ablations(datasets: list[str]) -> int:
    """Q9.1 — headline metrics paired with their serving-unavailable ablation."""
    import json

    from newsrec.eval import ablations

    out = ablations.run(datasets=tuple(datasets))
    print(ablations.format_table(json.loads(out.read_text())))
    print(f"\nwrote {config.repo_relative(out)}")
    return 0


def _run_eval(datasets: list[str]) -> int:
    """Q4 — ranking and beyond-accuracy metrics, sliced, with bootstrap CIs."""
    import json

    from newsrec.eval import harness

    out = harness.run(datasets=tuple(datasets))
    print(harness.format_table(json.loads(out.read_text())))
    print(f"\nwrote {config.repo_relative(out)}")
    return 0


def _run_semantic(datasets: list[str]) -> int:
    """Q3.5 — user vectors, ANN retrieval, semantic recall@K."""
    import json

    from newsrec.retrieval import semantic

    out = semantic.run(datasets=tuple(datasets))
    report = json.loads(out.read_text())
    print(semantic.format_table(report))
    print(f"\nimpression set matches BM25: {report['impression_set_matches_bm25']}")
    print(f"wrote {config.repo_relative(out)}")
    return 0


def _run_embed_diagnostics(datasets: list[str]) -> int:
    """How crowded the embedding space is (design_rationale 1.18)."""
    import json

    from newsrec.retrieval import embed

    out = embed.run_diagnostics(datasets=tuple(datasets))
    print(embed.format_diagnostics(json.loads(out.read_text())))
    print(f"\nwrote {config.repo_relative(out)}")
    return 0


def _run_embed(datasets: list[str], device: str, force: bool = False) -> int:
    """Q3.1, Q3.2 — encode every article, persist unit-norm vectors."""
    from newsrec.retrieval import embed

    summaries = embed.run(datasets=tuple(datasets), device=device, force=force)
    for name, s in summaries.items():
        if s.get("reused"):
            print(
                f"{name}: {s['embedded']:,} articles x {s['dim']}d reused "
                f"(fingerprint matches) -> {s['path']}"
            )
            continue
        print(
            f"{name}: {s['embedded']:,}/{s['articles']:,} articles x {s['dim']}d "
            f"on {s['device']} in {s['seconds']:.1f}s "
            f"({1000 * s['seconds'] / max(1, s['embedded']):.2f} ms/article) -> {s['path']}"
        )
    return 0


def _run_bm25_ablation(datasets: list[str]) -> int:
    """Q2.9 — the same recall measurement, once per tokenisation variant.

    Builds its own index per variant inside `evaluate_dataset`, so it never
    reads the headline index off disk: an index built under one tokeniser and
    scored under another would produce a plausible number from mismatched
    postings.
    """
    import json

    from newsrec.retrieval import ablation

    out = ablation.run(datasets=tuple(datasets))
    print(ablation.format_table(json.loads(out.read_text())))
    print(f"\nwrote {config.repo_relative(out)}")
    return 0


def _run_bm25(datasets: list[str]) -> int:
    """Q2 — build the index and measure recall@K.

    The index is rebuilt here rather than loaded, so `make bm25` is a genuine
    from-the-store rebuild and cannot report a number derived from a stale
    artefact on disk (Q1.5, Q7.2).
    """
    import json

    from newsrec.retrieval import index as idx
    from newsrec.retrieval import recall

    for dataset in datasets:
        dest = idx.index_dir(dataset)
        spec = config.tokeniser_for(dataset)
        from newsrec.data import feature_store

        articles = feature_store.read_table(
            config.FEATURE_STORE_DIR / dataset / "article_features.parquet"
        ).select("article_id", "text")
        built = idx.build(articles, spec)
        idx.save(built, dest)
        print(
            f"{dataset}: index {built.n_docs:,} docs / {len(built.vocab):,} terms / "
            f"{built.n_postings:,} postings -> {config.repo_relative(dest)}"
        )

    out = recall.run(datasets=tuple(datasets))
    print()
    print(recall.format_table(json.loads(out.read_text())))
    print(f"\nwrote {config.repo_relative(out)}")
    return 0


def _run_behaviour_features(datasets: list[str]) -> int:
    """A2 Q1 — summarise the click-history, session, and article behavioural
    features `make data` already built; writes nothing new to the store."""
    import json

    from newsrec.data import behaviour_features

    out = behaviour_features.run(datasets=tuple(datasets))
    print(behaviour_features.format_table(json.loads(out.read_text())))
    print(f"\nwrote {config.repo_relative(out)}")
    return 0


def _run_rerank(datasets: list[str]) -> int:
    """A2 Q2 — train GBDT/MLP re-rankers and ablate against the
    retrieval-only baseline. Trains on GPU; see rerank/mlp.py and
    rerank/gbdt.py for why there is no CPU fallback."""
    import json

    from newsrec.rerank import ablation

    out = ablation.run(datasets=tuple(datasets))
    print(ablation.format_table(json.loads(out.read_text())))
    print(f"\nwrote {config.repo_relative(out)}")
    return 0


def _run_nrms(requested: list[str] | None) -> int:
    """A2 Q3 — reproduce an NRMS-style ranker, then one principled change,
    ablated with paired bootstrap CIs. Trains on GPU, no CPU fallback.

    Takes the RAW `--dataset` flags rather than `main`'s resolved list, which
    defaults to every registered dataset. With no flag this stage defaults to
    `config.HEADLINE_DATASETS` instead — one English corpus and one Danish
    satisfies "both datasets", and training two neural variants on a third is
    a third more GPU time for a corpus A1 designated a fixture. See
    `nrms_ablation.run`."""
    import json

    from newsrec.rerank import nrms_ablation

    out = nrms_ablation.run(datasets=tuple(requested) if requested else None)
    print(nrms_ablation.format_table(json.loads(out.read_text())))
    print(f"\nwrote {config.repo_relative(out)}")
    return 0


def _run_rerank_eval(requested: list[str] | None) -> int:
    """A2 Q5 — the full metric set (including diversity, novelty and coverage),
    sliced, with bootstrap CIs, for the two-stage pipeline.

    Loads the models `nrms` trained rather than retraining, so the evaluated
    system is provably the one the ablation selected. Same default corpora as
    `nrms` for the same reason."""
    import json

    from newsrec.rerank import evaluate

    out = evaluate.run(datasets=tuple(requested) if requested else None)
    print(evaluate.format_table(json.loads(out.read_text())))
    print(f"\nwrote {config.repo_relative(out)}")
    return 0


def _run_serving(requested: list[str] | None) -> int:
    """A2 Q4 — measure the serving characteristics: index memory, p99 latency
    for a single user request with a per-stage breakdown, cost per 1000
    queries, and what binds first at 10x.

    Runs on CPU by design; see eval/serving.py for why that does not
    contradict the GPU-only training rule."""
    import json

    from newsrec.eval import serving

    out = serving.run(datasets=tuple(requested) if requested else None)
    print(serving.format_table(json.loads(out.read_text())))
    print(f"\nwrote {config.repo_relative(out)}")
    return 0


def _run_two_stage_ablation(requested: list[str] | None) -> int:
    """A2 Q9.1 — how much would the forbidden features be worth to the
    RE-RANKER? A1's ablations.json answers the same question for retrieval.

    Trains on GPU (it fits an MLP twice per dataset), so this is an a2-all
    stage rather than an `all` one."""
    import json

    from newsrec.rerank import unavailable

    out = unavailable.run(datasets=tuple(requested) if requested else None)
    print(unavailable.format_table(json.loads(out.read_text())))
    print(f"\nwrote {config.repo_relative(out)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    datasets = args.datasets or sorted(config.DATASETS)

    if args.stage == "data":
        return _run_data(datasets)
    if args.stage == "bm25":
        if args.ablation:
            return _run_bm25_ablation(datasets)
        return _run_bm25(datasets)
    if args.stage == "embed":
        if args.diagnostics:
            return _run_embed_diagnostics(datasets)
        return _run_embed(datasets, args.device, args.force)
    if args.stage == "ann":
        return _run_ann(datasets, args.measure)
    if args.stage == "semantic":
        return _run_semantic(datasets)
    if args.stage == "compare":
        return _run_compare(datasets)
    if args.stage == "eval":
        return _run_eval(datasets)
    if args.stage == "hybrid":
        return _run_hybrid(datasets)
    if args.stage == "ablations":
        return _run_ablations(datasets)
    if args.stage == "behaviour_features":
        return _run_behaviour_features(datasets)
    if args.stage == "rerank":
        return _run_rerank(datasets)
    if args.stage == "nrms":
        return _run_nrms(args.datasets)
    if args.stage == "rerank_eval":
        return _run_rerank_eval(args.datasets)
    if args.stage == "serving":
        return _run_serving(args.datasets)
    if args.stage == "two_stage_ablation":
        return _run_two_stage_ablation(args.datasets)
    if args.stage == "submit" and args.baseline:
        return _run_submit_baseline()
    if args.stage == "submit" and args.hybrid:
        return _run_submit_hybrid()
    if args.stage == "submit" and args.semantic:
        return _run_submit_semantic()

    _, reqs, doc = STAGES[args.stage]
    raise NotImplementedError(
        f"Stage '{args.stage}' ({reqs}) is not implemented yet.\n"
        f"  datasets: {', '.join(datasets)}\n"
        f"  plan:     agent_docs/{doc}\n"
        f"  checklist: agent_docs/assignment_requirements.md\n"
        "This stage fails loudly on purpose — a silent no-op would let the next "
        "stage read an empty artefact and report a plausible zero."
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
