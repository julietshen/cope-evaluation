# sentinel-grooming-eval

An independent evaluation of [Roblox Sentinel](https://github.com/Roblox/Sentinel)
for detecting grooming in **long multi-turn conversations**, plus a practical
implementation guide. This lives here (not in the Sentinel repo) because it is
an external assessment, not part of the library.

- **[REPORT.md](REPORT.md)** — findings: strong recall/ranking (AUC ≥ 0.95 up
  to 400-message conversations, 30/30 real predator logs flagged early), but
  the default skewness aggregator has serious failure modes and real-world
  precision cannot be measured with existing data.
- **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** — how to actually
  deploy Sentinel: index construction, per-sender buffering, windowing,
  aggregator choice, threshold calibration, review funnel, operational cautions.
- **`run_eval.py` / `dilution_test.py`** — the reproducible harness. Because
  the shipped synthetic dataset has no long conversations (and its
  conversation-level expert annotations reference files that were never
  published), the harness *composes* long conversations from expert-labeled
  single lines: innocuous background with grooming lines injected at
  controlled length, density, risk tier, and onset.
- **`results/`** (not committed) — running the scripts regenerates all CSV
  outputs and per-message score arrays behind REPORT.md in ~30 s. Kept out of
  git because derived outputs reference dataset conversations by name.

Requires the Sentinel repo (library + synthetic dataset) and optionally the
Perverted Justice dataset; see the reproduce section at the end of the guide.
