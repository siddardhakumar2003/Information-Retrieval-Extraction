"""newsrec — CS4.406 news recommendation on MIND and EB-NeRD.

Assignment 1 built lexical and semantic retrieval; Assignment 2 adds
behavioural signals from the click-logs and a re-ranking stage on top.

Pipeline stages, in dependency order:

    data      raw files -> unified schema -> temporal split -> feature store  [Q1]
              plus the click-history, session and article behavioural
              features the re-ranker consumes                             [A2.Q1]
    retrieval BM25 inverted index [Q2] and embedding + ANN retrieval [Q3]
    rerank    candidate-level features, GBDT/MLP and NRMS-style rankers,
              ablated against the retrieval-only baseline          [A2.Q2-Q3]
    eval      metrics, beyond-accuracy, slices, bootstrap CIs                 [Q4]
    submit    Codabench prediction files for both competitions                [Q5]

Stage 1 (retrieval) proposes candidates; stage 2 (rerank) re-orders the
impression's own candidate list. Nothing in `rerank` filters candidates out —
it only changes their order.

Configuration lives in `newsrec.config` and nowhere else.
"""

__version__ = "0.1.0"
