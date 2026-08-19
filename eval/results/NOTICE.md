# Notice — generated artifacts

Every file in this directory is a **machine-generated evaluation artifact produced by Claude (Claude Code, Opus 4.8)** working with Juliet Shen, via the `eval/` harness in this repository:

- `predictions_*.csv` — per-row model predictions and raw outputs
- `summary_*.csv` — per-policy metric tables
- `run_*.log`, `deploy_*.log` — run and deployment logs

These are model-conditioned outputs from third-party classifiers (cope-a/b, Shieldstral, gpt-oss-safeguard) scored by scripts in `eval/`. They reflect the policies, test sets, and prompt formats used at generation time and should be independently verified before any external or published use. The prediction CSVs are consumed by `eval/steerability.py`, `eval/offtopic.py`, and `eval/carveout.py`, so their column schema is intentionally left unmodified (no inline provenance headers) — this NOTICE covers their provenance instead.
