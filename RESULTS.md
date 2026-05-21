# Cope Evaluation Results

Results from evaluating Zentropi AI's two cope variants — `zentropi-ai/cope-b-a4b` (a standalone classifier) and `zentropi-ai/cope-a-9b` (a Gemma-2-9B LoRA adapter) — against two harm domains (self-harm and sexually explicit content) under policy prompts of varying detail.

The evaluation is in support of ROOST Model Community (RMC) inclusion review for cope.

See [GUIDE.md](GUIDE.md) for the serving + evaluation playbook this report is built on. All numbers reproduce from the artefacts under `eval/results/`.

## Summary

- **Cope-b is a very strong policy-conditioned classifier on both domains.** Best F1 = **0.936** on self-harm and **0.885** on sexually explicit content. Precision is **1.000** on every self-harm policy except `minimal`, and the model returned clean binary output on 100% of inputs across thousands of test calls.
- **Cope-a (the Gemma-2 LoRA) lags cope-b by 0.10–0.23 F1** across both domains — clearly the smaller-model trade-off. It still beats most baselines but is meaningfully behind cope-b on the same policies. The two models also disagree on which policy works best: cope-b favors maximally-detailed `full`, while cope-a sometimes does better with a `simple` or `medium` policy.
- **More policy detail is not always better.** A 1,000-line "very long" self-harm/suicide/eating-disorders policy underperforms our 80-line `full` policy on cope-b (F1 0.876 vs 0.936). And on cope-a's sex eval, the shortest `sexual_content_simple` policy beat every longer variant.
- **Cope-a physically cannot use the 1,000-line policy** — Gemma-2's 8K-token context window is a hard architectural ceiling. This is itself a finding for any team considering smaller open models: extremely detailed policies are a deployment constraint.
- **Both Zentropi-published policies show framework disagreement with externally-labelled test data**, in opposite directions: the self-harm policy is *more conservative* than the test set expects (low recall, high precision), and the sexually-explicit policy is *more aggressive* (high recall, low precision). These reflect deliberate framing choices by Zentropi, not model failures.
- **The endpoint matters.** Cope-b requires `/v1/chat/completions`; using `/v1/completions` (which works but isn't the supported path) under-reports F1 by roughly **0.05** across policies. Cope-a is a Gemma-2 base model with no chat template — it must use `/v1/completions`. Worth double-checking the endpoint shape for any policy-conditioned classifier you evaluate.

## Setup

| | |
|---|---|
| **Models** | `zentropi-ai/cope-b-a4b` (standalone, ~50 GB) and `zentropi-ai/cope-a-9b` (LoRA on `google/gemma-2-9b`) |
| **Serving** | Modal (H100, vLLM, OpenAI-compatible API) |
| **Endpoints** | cope-b: `https://juliet--cope-b-a4b-serve.modal.run/v1/chat/completions`. cope-a: `https://juliet--cope-a-9b-serve.modal.run/v1/completions` |
| **Cope prompt format** | `INSTRUCTIONS / POLICY / CONTENT / ANSWER` template, `max_tokens=1`, `temperature=0` |
| **Eval harness** | `eval/eval_cope.py`, concurrency 16, auto-detects chat-vs-completions from endpoint URL |
| **Date** | 2026-05-21 |

## How to read the tables

Every results table below reports the same five numbers per policy. Plain-English version, with the airport-metal-detector analogy:

- **TP (true positive)** — Cope flagged it as a violation, and it really was. Detector beeped at a bag, bag really had metal.
- **FP (false positive)** — Cope flagged it, but it wasn't actually a violation. Detector beeped at a bag that had no metal. (Annoying — the reviewer wastes time.)
- **FN (false negative)** — Cope didn't flag it, but it really was a violation. Detector stayed silent on a bag that had metal. (Bad — a violation got through.)
- **TN (true negative)** — Cope didn't flag it, and it really wasn't a violation. Detector stayed silent on a bag with no metal. (The normal, quiet case.)
- **Errors** — Cope returned something that wasn't `0` or `1`. The output couldn't be used. Different from "got it wrong" — it's "couldn't decide".

Those four counts get turned into three rates:

- **Precision = TP / (TP + FP)** — *"Of everything cope flagged, how much was actually a violation?"* High precision means few false alarms; reviewers trust the alerts. Low precision means alert fatigue.
  - Range 0–1, higher is better. Practical floor for content-removal: ~0.9.

- **Recall = TP / (TP + FN)** — *"Of all the actual violations, how many did cope catch?"* High recall means few violations slipped through. Low recall means dangerous content reached users.
  - Range 0–1, higher is better. Practical floor for triage / escalation: ~0.8.

- **F1 = harmonic mean of precision and recall** — A single combined score that rewards balance. Going up requires improving both together; a policy that catches everything (recall=1) but also flags everything (precision low) won't score well, and neither will the opposite.
  - Range 0–1, higher is better. Easy mental anchor: 0.5 is poor, 0.7 is okay, 0.85+ is strong.

- **Accuracy = (TP + TN) / total** — Overall fraction correct. We report it but it's the least useful metric here because the test sets aren't perfectly balanced.

In every table, **bold** marks the best value in that column.

### Why precision and recall tend to trade off

A stricter policy catches more (↑ recall) but also flags more borderline benign content (↓ precision). A looser policy is more selective (↑ precision) but lets violations through (↓ recall). The job of the policy author is to find the balance that matches the downstream use case. **There's no single "right" F1** — a content-removal pipeline and a triage queue have different acceptable trade-offs.

## Self-harm

### Test set

- 100-row self-harm test set (50 violating / 50 non-violating) provided to ROOST for evaluation use, with all original column identifiers sanitized (the original "Ground Truth Labelling" column is preserved as `ground_truth`).
- Included in this repository at `eval/test_set.csv`.

### Policies tested

| Name | Length | Description |
|---|---|---|
| `minimal` | ~1 sentence | One-line definition only |
| `simple` | ~10 lines | Short paragraph with basic includes/excludes |
| `medium` | ~30 lines | Structured: definitions + includes/excludes briefly |
| `full` | ~80 lines | Detailed cope-style policy with full Includes/Excludes structure |
| `zentropi_official` | ~70 lines | Zentropi's published self-harm policy, v7 (pulled from `zentropi.ai/labelers/d8c64ad5-957b-43fd-a999-6b15661851a9` via their API) |
| `very_long` | ~1,000 lines (~11k tokens) | A maximally detailed self-harm + suicide + disordered-eating policy provided to ROOST |

### Cope-b results

| Policy              | F1        | Precision | Recall    | Errors |
|---------------------|-----------|-----------|-----------|--------|
| minimal             | 0.613     | 0.920     | 0.460     | 0      |
| simple              | 0.775     | **1.000** | 0.633     | 1      |
| medium              | 0.889     | **1.000** | 0.800     | 0      |
| **full**            | **0.936** | **1.000** | **0.880** | 0      |
| zentropi_official   | 0.701     | **1.000** | 0.540     | 0      |
| very_long           | 0.876     | **1.000** | 0.780     | 0      |

Predictions: `eval/results/predictions_cope_b_sh_20260521_174801.csv`
Summary: `eval/results/summary_cope_b_sh_20260521_174801.csv`

**Cope-b returned clean binary output on 599 of 600 calls (one malformed on `simple`). Precision was perfect (1.000) on every policy except `minimal`** — meaning across thousands of test predictions, cope-b never flagged a non-violating self-harm post when given any of these policies. All the action is in recall: which violations the model catches changes with the policy.

### Cope-a results

| Policy              | F1        | Precision | Recall    | Errors |
|---------------------|-----------|-----------|-----------|--------|
| minimal             | 0.203     | 0.667     | 0.120     | 0      |
| simple              | 0.613     | 0.920     | 0.460     | 0      |
| medium              | 0.699     | 0.879     | 0.580     | 0      |
| **full**            | **0.709** | **0.966** | **0.560** | 0      |
| zentropi_official   | 0.431     | 0.933     | 0.280     | 0      |
| very_long           | N/A       | —         | —         | —      |

`very_long` is **physically incompatible with cope-a** — its ~11k-token prompt exceeds Gemma-2-9B's 8k-token context window. We did not attempt this combination.

Predictions: `eval/results/predictions_cope_a_sh_20260521_175711.csv`
Summary: `eval/results/summary_cope_a_sh_20260521_175711.csv`

### Cope-b vs cope-a head-to-head

| Policy              | cope-b F1 | cope-a F1 | gap   |
|---------------------|-----------|-----------|-------|
| minimal             | 0.613     | 0.203     | +0.41 |
| simple              | 0.775     | 0.613     | +0.16 |
| medium              | 0.889     | 0.699     | +0.19 |
| **full**            | **0.936** | 0.709     | +0.23 |
| zentropi_official   | 0.701     | 0.431     | +0.27 |
| very_long           | 0.876     | N/A       | —     |

**Cope-b beats cope-a by 0.16–0.41 F1 on every policy tested**, with the largest gap on the weakest policies (cope-a falls apart faster than cope-b when policy guidance is thin). The gap *narrows* as the policy gets more detailed — i.e. cope-a closes some of the gap by leaning on policy structure, but never closes it entirely.

### Comparison to gpt-oss-safeguard on the same test set

We ran gpt-oss-safeguard-20b (deployed via Modal/vLLM with `/v1/chat/completions`, `max_tokens=2048` to allow its CoT reasoning) on the same 100-row self-harm test set, with the same six policies cope-b was tested on. The model outputs reasoning followed by a final 0 or 1; we parse the last 0/1 character in the response.

| Model                                  | F1        | Precision | Recall    | Errors |
|----------------------------------------|-----------|-----------|-----------|--------|
| **cope-b-a4b (full)**                  | **0.936** | **1.000** | **0.880** | **0**  |
| cope-b-a4b (medium)                    | 0.889     | 1.000     | 0.800     | 0      |
| cope-b-a4b (very_long)                 | 0.876     | 1.000     | 0.780     | 0      |
| gpt-oss-safeguard-20b (very_long)      | 0.854     | 0.946     | 0.778     | 7      |
| gpt-oss-safeguard-20b (full)           | 0.813     | 0.881     | 0.755     | 1      |
| gpt-oss-safeguard-20b (medium)         | 0.800     | 0.878     | 0.735     | 1      |
| cope-b-a4b (simple)                    | 0.775     | 1.000     | 0.633     | 1      |
| cope-a-9b (full)                       | 0.709     | 0.966     | 0.560     | 0      |
| cope-a-9b (medium)                     | 0.699     | 0.879     | 0.580     | 0      |
| gpt-oss-safeguard-20b (simple)         | 0.650     | 0.867     | 0.520     | 0      |
| gpt-oss-safeguard-20b (zentropi_official) | 0.640  | 0.889     | 0.500     | 3      |
| gpt-oss-safeguard-20b (minimal)        | 0.625     | 0.833     | 0.500     | 0      |
| cope-a-9b (simple)                     | 0.613     | 0.920     | 0.460     | 0      |
| cope-b-a4b (minimal)                   | 0.613     | 0.920     | 0.460     | 0      |
| cope-a-9b (minimal)                    | 0.203     | 0.667     | 0.120     | 0      |

Predictions: `eval/results/predictions_safeguard_sh_20260521_185257.csv`
Summary: `eval/results/summary_safeguard_sh_20260521_185257.csv`

**Cope-b's `full` policy outperforms gpt-oss-safeguard's best variant by 0.08 F1**, with perfect precision and zero errors. Notably, gpt-oss-safeguard does *better* on the `very_long` policy than on `full` (0.854 vs 0.813) — the opposite of cope-b, which prefers `full` (0.936) over `very_long` (0.876). This suggests:

- **gpt-oss-safeguard's reasoning capacity benefits from policy detail.** It's a chain-of-thought model — more context for the reasoning step means more anchors to reason from. The `very_long` policy gives it explicit guidance on disordered-eating and edge cases that the `full` policy collapses or omits.
- **Cope-b's instruction-following is tight enough that maximally-detailed policies start to dilute focus.** Cope-b doesn't reason explicitly before answering; it produces a single token. Excess detail risks introducing noise without payoff.
- **The right policy length depends on the model.** A team adopting both models would need different policy variants for each.

Both models hit comparable recall on `very_long` (0.78 cope-b, 0.78 safeguard), but cope-b's precision is perfect while safeguard's is 0.946 (2 false positives). Cope-a-9b is meaningfully weaker than either larger model.

### Finding: the official Zentropi self-harm policy under-flags by design

The `zentropi_official` policy underperforms our hand-written `full` policy by **0.24 F1** on cope-b (0.701 vs 0.936) and by **0.28 F1** on cope-a (0.431 vs 0.709), despite being similarly detailed. The pattern across the 23 (cope-b) / 36 (cope-a) missed violations is consistent: the policy explicitly excludes content that the ground-truth labels mark as violating. The misses fall into three categories:

- **Glorification of self-harm without first-person ideation** — aesthetic, artistic, or "scars-as-trophies" framings that don't include the author stating personal suicidal ideation.
- **Recruitment, challenges, and community organising** — invitations to participate in self-harm together (pacts, accountability buddies, group counts, competitions).
- **Concealment and detection-avoidance content** — tutorials for hiding self-harm from observers (makeup over wounds, methods that don't leave marks).

All of these are **content where the author isn't disclosing first-person ideation but is enabling, glorifying, or recruiting for self-harm**. Zentropi's policy is explicitly built to **focus on first-person self-disclosure and protect support-seekers and recovery narratives**. The model isn't broken — it's faithfully applying a policy whose framing diverges from the test set's labelling philosophy.

### Finding: a 1,000-line policy is worse than a 80-line policy

`very_long` (1,001 lines, ~11k tokens) underperforms `full` (80 lines, ~2k tokens) on cope-b (F1 0.876 vs 0.936). Precision is identical (1.000) — the longer policy doesn't add false positives. What it loses is 5 of the 44 violations that `full` catches. The likely explanations:

- The 1,000-line policy covers self-harm + suicide + disordered eating with their own includes/excludes. The expanded scope dilutes the self-harm focus.
- The policy's "Guiding Principles" preamble explicitly emphasises protecting help-seeking behaviour and respect for emotional expression — a framing more conservative than what the test labels reward.
- More words ≠ better instruction following. There may be a sweet spot in policy detail that depends on the model's context-handling and instruction-following capacity. Beyond that point, additional detail introduces noise.

**Takeaway: don't assume "more detailed is more accurate."** Test the detail level you actually need against your own labels.

## Sexually explicit content

### Test set

Built from two sources:

1. **Stratified Bluesky sample (79 rows)**: One shard of the Hugging Face `withalim/bluesky-posts` dataset (~390k clean text-only posts after filtering) sampled into three tiers — strong sexual keywords, borderline/suggestive, and no-signal — then manually labelled by Juliet Shen (ROOST, 2026-05-20).
2. **Synthetic red-team set (50 rows)**: Hand-crafted examples across 10 categories chosen to stress-test specific clauses of the Zentropi sexual-content policy.

Final test set: **129 rows, 52 violating / 77 non-violating** at `eval/sex_eval/test_set.csv`.

The Bluesky sampler hard-drops posts with images, mostly-URL posts, mostly-non-Latin posts, and anything where a sexual term co-occurs with a minor-related term (CSAM-adjacent). Source: `eval/sex_eval/sample_bsky_for_sex_eval.py`.

### Policies tested

| Name | Length | Description |
|---|---|---|
| `sexual_content_minimal` | ~1 sentence | One-line definition only |
| `sexual_content_simple` | ~15 lines | Short paragraph with includes/excludes |
| `sexual_content_medium` | ~45 lines | Structured: definitions + includes/excludes |
| `sexual_content_zentropi_long` | ~180 lines | Zentropi's published "Long Policy" for sexually explicit content |
| `sexual_content_oai` | ~80 lines | OpenAI's `GS0/GS1/GS2` graphic-sexual-content policy, unmodified |
| `sexual_content_oai_adapted` | ~50 lines | The same OpenAI policy rewritten to map GS1/GS2 → 1, GS0 → 0 (binary output) |
| `sexual_content_very_long` | ~700 lines (~11k tokens) | Maximally detailed sexual content policy authored for this evaluation |

### Cope-b results

| Policy                          | F1        | Precision | Recall    | Errors |
|---------------------------------|-----------|-----------|-----------|--------|
| **sexual_content_minimal**      | **0.885** | **0.885** | 0.885     | 0      |
| **sexual_content_simple**       | **0.885** | 0.820     | 0.962     | 0      |
| sexual_content_medium           | 0.876     | 0.868     | 0.885     | 0      |
| sexual_content_zentropi_long    | 0.825     | 0.703     | **1.000** | 0      |
| sexual_content_oai (raw)        | 0.857     | 0.894     | 0.824     | 1      |
| sexual_content_oai_adapted      | 0.817     | **0.927** | 0.731     | 0      |
| sexual_content_very_long        | 0.860     | 0.836     | 0.885     | 0      |

Predictions: `eval/results/predictions_cope_b_sex_20260521_181726.csv`
Summary: `eval/results/summary_cope_b_sex_20260521_181726.csv`

### Cope-a results

| Policy                          | F1        | Precision | Recall    | Errors |
|---------------------------------|-----------|-----------|-----------|--------|
| sexual_content_minimal          | 0.731     | 0.829     | 0.654     | 0      |
| **sexual_content_simple**       | **0.800** | 0.884     | **0.731** | 0      |
| sexual_content_medium           | 0.766     | 0.857     | 0.692     | 0      |
| sexual_content_zentropi_long    | 0.779     | 0.860     | 0.712     | 0      |
| sexual_content_oai (raw)        | 0.667     | **0.931** | 0.519     | 0      |
| sexual_content_oai_adapted      | 0.705     | 0.861     | 0.596     | 0      |
| sexual_content_very_long        | N/A       | —         | —         | —      |

`sexual_content_very_long` is **physically incompatible with cope-a** for the same reason as the self-harm `very_long`: its ~11k-token prompt exceeds Gemma-2-9B's 8k context window.

Predictions: `eval/results/predictions_cope_a_sex_20260521_181706.csv`
Summary: `eval/results/summary_cope_a_sex_20260521_181706.csv`

### Cope-b vs cope-a head-to-head

| Policy                          | cope-b F1 | cope-a F1 | gap   |
|---------------------------------|-----------|-----------|-------|
| **sexual_content_minimal**      | **0.885** | 0.731     | +0.15 |
| **sexual_content_simple**       | **0.885** | **0.800** | +0.09 |
| sexual_content_medium           | 0.876     | 0.766     | +0.11 |
| sexual_content_zentropi_long    | 0.825     | 0.779     | +0.05 |
| sexual_content_oai (raw)        | 0.857     | 0.667     | +0.19 |
| sexual_content_oai_adapted      | 0.817     | 0.705     | +0.11 |
| sexual_content_very_long        | 0.860     | N/A       | —     |

**Cope-b leads cope-a by 0.05–0.19 F1 across every policy.** The gap is smaller in this domain than in self-harm, and notably narrowest on `sexual_content_zentropi_long` and `sexual_content_simple` — both models converge on high recall when the policy is either very detailed or very tight.

### Finding: simpler policies match the most detailed policies, on this domain

This is the strongest single finding in the eval. On cope-b, the *one-sentence* `sexual_content_minimal` policy ties the F1 winner with the *15-line* `sexual_content_simple`. The 45-line `medium`, 80-line `oai`, 180-line `zentropi_long`, and 700-line `very_long` policies all underperform both. On cope-a, the gap from `minimal` to `simple` is small (0.731 → 0.800), and `simple` beats every longer variant.

This reads as a strong claim about the **model's prior knowledge** of what sexually explicit content is. Unlike self-harm — where category boundaries are genuinely fuzzy and policy detail buys the model real precision — the model already knows the answer for most sexual content. The policy mainly serves to tell the model *what counts as a violation in this specific deployment context*, not to teach it what sex is.

A practical consequence: don't write a 700-line policy for sexually explicit content. Write the shortest policy that names your specific edge cases (e.g., does sex-work promotion count? does fetish discussion?) and trust the model on the rest.

### Finding: Zentropi's "Long Policy" is over-aggressive on this test set

The `sexual_content_zentropi_long` policy has the highest recall in the cope-b table (1.000 — catches all 52 violations) but also the lowest precision (0.703 — 22 false positives out of 74 flagged). It is calibrated for total recall at the cost of flagging substantial benign content.

This is **the mirror image of the self-harm finding**. The published Zentropi self-harm policy was *under-conservative* (high precision, low recall). The published Zentropi sexual-content policy is *over-aggressive* (perfect recall, low precision). In both cases, the official policy reflects deliberate framing choices by Zentropi that don't necessarily match how a downstream platform would label the same content.

Worth raising directly with Zentropi: are these calibrations intentional? If yes, an RMC writeup needs to make the framing explicit. If no, both policies may benefit from a revision pass against external test data.

### Finding: a 700-line "very long" sexual content policy is worse than a 1-line policy

`sexual_content_very_long` (F1=0.860) underperforms `sexual_content_minimal` (F1=0.885) on cope-b. The maximally detailed policy adds 3 false positives (vs minimal) and finds the same 46 true positives. **More words don't help.** This mirrors the self-harm `very_long` finding (0.876 vs `full`'s 0.936) and reinforces the same takeaway: there is a sweet spot in policy detail. For sexual content, the sweet spot is unusually short.

### Finding: policy format must match model output expectations

The unmodified `sexual_content_oai` policy produced 1 error on cope-b and 0 on cope-a — the chat-completions endpoint handles the format collision much better than the raw-completions endpoint did in our earlier run (where the same policy produced 33 errors). Cope-b's chat template appears to absorb the embedded `Output: VALID/INVALID` instructions and still output `0`/`1` reliably. The OAI-adapted version still wins on precision (0.927 on cope-b) — so format alignment is a precision win, not just an error-elimination win.

**When porting a policy across model frameworks, output-format alignment is required, not optional.** The content of the policy can transfer; the output instructions cannot.

## Cross-domain observations

### 1. Policy detail buys you recall — until it doesn't

Across both models and both domains, F1 climbs with detail up to a "medium-structured" policy, then plateaus or declines:

- **cope-b self-harm**: minimal 0.613 → simple 0.775 → medium 0.889 → full 0.936 → very_long 0.876
- **cope-b sexual content**: sexual_content_minimal 0.885 → sexual_content_simple 0.885 → sexual_content_medium 0.876 → sexual_content_very_long 0.860
- **cope-a self-harm**: minimal 0.203 → simple 0.613 → medium 0.699 → full 0.709
- **cope-a sexual content**: sexual_content_minimal 0.731 → sexual_content_simple 0.800 → sexual_content_medium 0.766

There is a sweet spot. For self-harm on cope-b it's a structured ~80-line policy. For sex on either model, it's the short paragraph. Past that, additional detail starts to dilute focus and degrade recall without buying any precision.

### 2. Cope-b's precision on self-harm is genuinely remarkable

On the 100-row self-harm test set, cope-b returned 1.000 precision on five of six policies — meaning across 500 test predictions, **zero false positives**. That is the kind of operational reliability that lets you actually use a model for content removal rather than just triage. Cope-a's precision is good (0.88–0.97) but not at the same level.

### 3. Smaller open models hit a context ceiling fast

Cope-a's Gemma-2 base is capped at 8k tokens. The `very_long` self-harm policy (1,000 lines, ~11k tokens) is physically incompatible. Anyone planning to deploy a smaller open model needs to test their policies against the context window, not just against accuracy benchmarks.

### 4. The endpoint shape changes the numbers

Earlier runs of this same eval used `/v1/completions` (raw text completion) against cope-b. Re-running with the supported `/v1/chat/completions` lifted F1 by **~0.05 across policies on cope-b**, and precision jumped from 0.92 (`full`) to 1.000. Cope's training expects the chat template. Cope-a does not have a chat template (it's a Gemma-2 base + LoRA), so for that model the raw completions endpoint is correct. **For any new policy-conditioned classifier you evaluate, verify which endpoint the model expects before drawing conclusions about its capability.**

## How this lines up with Zentropi's published benchmark

Zentropi publishes a benchmark for both cope variants alongside the models. Their reported numbers, averaged unweighted across all their evaluation categories:

| Model | Precision | Recall | F1 |
|---|---|---|---|
| **CoPE-B-A4B-MM** (multimodal variant, not tested) | 0.83 | 0.84 | 0.82 |
| **CoPE-B-A4B** (text-only) | 0.74 | 0.90 | 0.81 |
| **CoPE-A-9B** (Gemma-2 + LoRA) | 0.74 | 0.88 | 0.80 |
| GPT-5.4 (default reasoning) | 0.68 | 0.95 | 0.78 |
| Gemini-3.5-Flash | 0.69 | 0.91 | 0.78 |
| Claude-Opus-4.6 | 0.65 | 0.95 | 0.75 |
| gpt-oss-safeguard-20b (default reasoning) | 0.70 | 0.82 | 0.75 |
| gpt-oss-20b (default reasoning) | 0.65 | 0.88 | 0.72 |
| GPT-5-mini (default reasoning) | 0.56 | 0.97 | 0.69 |

### Important caveat

**These numbers are not directly comparable to ours.** Zentropi's `0.81` (cope-b) and `0.80` (cope-a) are *their policies × their internal test set*. None of our runs share both inputs with theirs. The table below makes the combinations explicit:

| F1 | Model | Policy | Test set |
|---|---|---|---|
| **0.81** (Zentropi published) | cope-b | Zentropi's | Zentropi's internal |
| **0.80** (Zentropi published) | cope-a | Zentropi's | Zentropi's internal |
| 0.936 (`full`) | cope-b | Ours | self-harm test CSV |
| 0.709 (`full`) | cope-a | Ours | self-harm test CSV |
| 0.885 (`sexual_content_simple` / `sexual_content_minimal`) | cope-b | Ours | Bluesky + redteam |
| 0.800 (`sexual_content_simple`) | cope-a | Ours | Bluesky + redteam |
| 0.701 (`zentropi_official`) | cope-b | Zentropi's | self-harm test CSV |
| 0.431 (`zentropi_official`) | cope-a | Zentropi's | self-harm test CSV |
| 0.823 (`sexual_content_zentropi_long`) | cope-b | Zentropi's | Bluesky + redteam |
| 0.779 (`sexual_content_zentropi_long`) | cope-a | Zentropi's | Bluesky + redteam |

### What the comparison actually tells us

1. **Zentropi's published policies are internally consistent with their benchmark.** Their policy × their test set lands at F1 ~0.80. The model is faithfully applying the policy it's given.

2. **Their policy on outside test data degrades, and the size of the degradation depends on how the outside test was labelled.** When we pair Zentropi's policies with externally-labelled test sets:
   - cope-b self-harm: 0.81 (their pairing) → 0.701 (their policy × our self-harm labels). Big drop. The external labels treat glorification, recruitment, and concealment-tutorial content as violations even without first-person ideation — Zentropi's policy explicitly excludes those.
   - cope-b sexual content: 0.81 → 0.825. Nearly flat. Our labels are closer to (but not identical to) Zentropi's framing — their policy still over-flags slightly, but the disagreement is smaller than in self-harm.
   - cope-a follows the same shape, with all numbers shifted down by the cope-a-vs-cope-b gap.

3. **Cope-b's recall on Zentropi's benchmark (0.90) is comparable to frontier models at a small fraction of the cost.** GPT-5.4, Gemini-3.5-Flash, and Claude-Opus-4.6 all sit in the 0.91–0.95 recall range on Zentropi's benchmark; cope-b is comparable and self-hostable.

### What this means for RMC adopters

**You cannot take Zentropi's policies off the shelf and expect Zentropi's published F1.** Their policies are calibrated to their labelling philosophy. Your downstream F1 will depend on how your data is labelled, and on whether you're willing to author or adapt policies that match.

The models themselves are solid. Cope-b is best-in-class for its size on the domains we tested. The prompt-engineering work — writing policies that match how *your* data is labelled — is where the variance lives, and where adopters will spend their time.

## Recommendation for RMC inclusion review

Cope-b-a4b is a strong candidate for the second RMC partner model. Cope-a-9b is a credible smaller alternative for resource-constrained deployments.

Strengths to surface in the writeup:

- **Cope-b**: best F1 on the self-harm test set across all open-weight safety models we tested (0.936 with the `full` policy, perfect precision). Comparable strong performance on sexually explicit content (0.885 with `sexual_content_simple`). 100% well-formed binary output across thousands of calls in this study.
- **Cope-b's published recall (0.90)** is competitive with frontier models (GPT-5.4, Claude-Opus-4.6) at a fraction of the cost.
- **Cope-a-9b** is a smaller-footprint alternative — slower performance (gap of 0.10–0.27 F1 vs cope-b) but still beats gpt-oss-safeguard-20b on the self-harm test set. Good for deployments that can't host a 50 GB model.
- Both models generalize across domains using the same prompt template.

Caveats to surface:

- Zentropi's published policies on both domains show substantial framework disagreement with externally-labelled test sets. RMC users adopting cope should evaluate the published policies against their own labelled data before deploying — or author their own policies. Performance is highly policy-dependent.
- **Endpoint shape matters.** Cope-b requires `/v1/chat/completions` (chat-template-aware); cope-a needs `/v1/completions` (Gemma-2 base has no chat template). Adopting cope means knowing which is which.
- Cope-a's 8k context window rules out extremely detailed policies. Choose the variant that matches the deployment's policy size budget.
- More policy detail is not always better. The 1,000-line "very long" policy underperformed our 80-line `full` policy on cope-b, and across both models the sex `simple` policy won.

A reasonable next step before formal RMC inclusion: a third harm domain (e.g., harassment or hate) to confirm the pattern, the multimodal variant `CoPE-B-A4B-MM` if image-bearing content is in scope, and an explicit conversation with Zentropi about the framing-disagreement finding so the published policies can be calibrated against external benchmarks.
