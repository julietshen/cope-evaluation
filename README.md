# cope-evaluation

Evaluation of `zentropi-ai/cope-b-a4b` against two harm domains — self-harm and sexually explicit content — under policy prompts of varying detail. Run in support of ROOST Model Community (RMC) inclusion review.

**Round 2 (August 2026):** `mistralai/Shieldstral-1.0-3B` run on the same test sets and policies, locally (no GPU rental). Best F1 0.922 (self-harm) / 0.852 (sexual content) — just behind cope-b at 1/16th the size. See the round-2 section of RESULTS.md.

## What's in here

- **[RESULTS.md](RESULTS.md)** — the findings. Per-policy precision/recall/F1, head-to-head against gpt-oss-safeguard, comparison against Zentropi's published benchmark, RMC inclusion recommendation. **Start here.**
- **[GUIDE.md](GUIDE.md)** — the general playbook this evaluation followed. Serving any open-weight policy classifier on Modal, building test sets, running the eval harness, adapting for a different model.
- `serve_cope.py` — Modal deployment recipe.
- `eval/` — the harness, policies, test sets, and per-run prediction/summary CSVs.

## Reproducing

Assumes you have a Modal account, an HF token with access to `zentropi-ai/cope-b-a4b`, and Python 3.11+.

```bash
# 1. Stand up the endpoint (Part 1 of GUIDE.md)
uv tool install modal
modal token new
modal secret create cope-secrets HF_TOKEN=hf_... VLLM_API_KEY=sk-pick-something
modal run serve_cope.py::download_model    # one-time, ~$0.01
modal deploy serve_cope.py

# 2. Run the eval (Part 2 of GUIDE.md)
export VLLM_API_KEY=sk-pick-something      # same value as above
pip install requests
cd eval
python eval_cope.py --policies minimal simple medium full zentropi_official
python eval_cope.py --test-set sex_eval/test_set.csv --label sex \
  --policies sexual_content_minimal sexual_content_simple sexual_content_medium sexual_content_zentropi_long sexual_content_oai sexual_content_oai_adapted
```

Expect ~$2 in Modal GPU time for both evals end-to-end.

The round-2 Shieldstral runs need no Modal at all — a venv with `torch` and `transformers>=5` on an Apple-silicon Mac (or any GPU box) is enough:

```bash
cd eval
python eval_shieldstral.py --label sh --policies minimal simple medium full very_long zentropi_official
python eval_shieldstral.py --test-set sex_eval/test_set.csv --label sex \
  --policies sexual_content_minimal sexual_content_simple sexual_content_medium \
  sexual_content_zentropi_long sexual_content_oai sexual_content_oai_adapted sexual_content_very_long
```

## Provenance and sanitization

- The **self-harm test set** (`eval/test_set.csv`) was provided to ROOST by an external partner for evaluation use. All labelling-org identifiers in the original column names have been replaced with neutral descriptors (`ground_truth`). Post content is preserved as-is.
- The **sexually-explicit Bluesky candidates** (`eval/sex_eval/candidates_to_label.csv`) are sampled from the public `withalim/bluesky-posts` Hugging Face dataset. Labels in that file are this author's editorial judgment (labeller: Juliet Shen, ROOST, 2026-05-20), not ground truth from a benchmark partner.
- The **red-team set** (`eval/sex_eval/redteam_set.csv`) is synthetic content hand-crafted to stress-test policy clauses. **NSFW**: contains explicit text by design.
- The **policy files** under `eval/policies/` are either (a) hand-written for this evaluation, (b) pulled from Zentropi's public labelers API, or (c) sourced from OpenAI's `teen-safety-policy-pack` repository.

## Status

This evaluation was conducted as part of an RMC inclusion review for cope-b-a4b in May 2026. Findings will be shared with Zentropi for response and with the wider RMC community.
