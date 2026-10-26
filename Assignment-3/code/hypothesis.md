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

*(Not yet started. Fill in the chosen property to vary, the settings to test, and a short
hypothesis here, commit it, then implement the generator + `scripts/part4_*.py`.)*

- **Property varied:**
- **Settings tested (>=3 plus a control):**
- **Hypothesis:**
- **Reasoning:**
