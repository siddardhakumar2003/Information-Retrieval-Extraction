## Prompt 1

*2026-08-18T14:17:22.306Z*

I'm building a lexical and semantic news recommendation project on the MIND and EB-NeRD datasets. The full spec is Assignment1_v1.pdf in this directory. Read it first.

Do not write any implementation code in this session. This is setup only. Produce a plan covering:

Requirements extraction — read the PDF and propose a checklist, with the deliverables.
Repo structure — package layout, where raw/intermediate/processed data lives, the feature store, prediction files, the eval harness, and a Makefile whose targets give a genuine one-command rebuild from raw files.

Tooling — dependency manager, linter/formatter, pytest layout, .gitignore, and a pre-commit hook blocking large files and data artefacts.

For Q1 — tell which tests the data pipeline will need before it's written: schema contracts, temporal-split ordering asserted on timestamps.

For every significant design decision in this plan, list the alternatives you considered and their trade-offs. Where the choice is genuinely contested, do not pick for me — present the options with your reasoning and let me decide.

Ask me clarifying questions where the spec is ambiguous or where my preferences matter (Python tooling, GPU availability, whether EB-NeRD Danish-language handling is in scope from the start, unified vs. parallel dataset schemas).

---

## Prompt 2

*2026-08-18T18:42:33.173Z*

I have downloaded the MIND and EB-NeRD datasets in the Data folder

---
