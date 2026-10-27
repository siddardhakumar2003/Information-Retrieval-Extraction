# Hypotheses (Parts 3 & 4)

Per the assignment: each experiment's hypothesis must be written down and **committed to git
before** that experiment is run, so the git history shows the hypothesis predates the results.

## Part 3: experiment on the standard data

- **Experiment chosen:** Scaling. On SIFT, repeat the Part 2 benchmark at 100K, 300K and 1M
  vectors, for exact search, HNSW and IVF-Flat (IVF-PQ excluded, per the assignment's wording
  for this option), and see how the cost of reaching recall@10 >= 0.95 grows with N.

- **Hypothesis:** Exact search's queries-per-second will degrade roughly *linearly* with N,
  since brute-force search compares the query against every stored vector — expect roughly a
  10x drop in QPS going from 100K to 1M vectors. HNSW's QPS at recall 0.95 will degrade *much
  more slowly*, closer to a log(N) relationship, since the number of graph hops a query needs
  grows only logarithmically with the number of indexed points. IVF-Flat, with `nlist` scaled
  at roughly sqrt(N) at each size (the same build-grid formula used in Part 2, so this isn't a
  methodology change — it's how nlist is already chosen), should land in between: its
  per-query cost (number of vectors scanned within the probed clusters) grows roughly with
  sqrt(N), worse than HNSW but far better than exact. Build time is expected to grow roughly
  linearly (or N log N) for HNSW and roughly linearly for IVF-Flat's k-means training + vector
  assignment step; exact search's "build" (just loading vectors into a flat index, no training)
  should stay cheap regardless of N.

- **Reasoning:** This is the standard algorithmic-complexity argument for why approximate
  nearest-neighbour search exists at all — sublinear query cost vs. brute force's linear cost —
  and this experiment is designed to make that trade-off directly measurable on real data
  rather than just assumed from big-O notation.

## Part 4: experiment on self-generated data

- **Property varied:** Dimensionality (d) of self-generated uniform-random synthetic data
  (`annbench.synthetic.generate_uniform`, points drawn i.i.d. uniform in `[-1, 1]^d`). Dataset
  size and random seed are held fixed — only `d` changes between settings. Tested on HNSW and
  IVF-Flat (exact search included as the unaffected reference), each at one fixed build+search
  configuration (not swept to a recall target, unlike Parts 1-3).

- **Settings tested (>=3 plus a control):** d = 32 (control), 64, 128, 256.

- **Hypothesis:** Recall@10 at a fixed HNSW `ef` and a fixed IVF-Flat `nprobe` will degrade
  monotonically as d increases from 32 to 256, because the ratio of nearest-to-farthest
  neighbour distance shrinks as dimensionality grows ("distance concentration" / the curse of
  dimensionality), making distance-based pruning less able to discriminate true near neighbours
  from distractors — both HNSW's graph search and IVF-Flat's cluster-probe pruning rely on
  exactly that discrimination. Exact search's recall should stay at 1.0 throughout, since
  brute-force search never prunes and only its absolute distances change, not its ranking.

- **Reasoning:** This is the standard, well-documented curse-of-dimensionality argument — the
  assignment names this exact phenomenon as an example ("with uniform random vectors, nearest
  neighbours become barely closer than everything else as the dimension grows"). High
  confidence of confirmation is expected here, which the assignment says is fine: "a hypothesis
  that turns out wrong is fine; we grade the reasoning and the care of the experiment."
