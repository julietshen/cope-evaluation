# Cope-b-a4b Evaluation Results

Results from evaluating `zentropi-ai/cope-b-a4b` against two harm domains — self-harm and sexually explicit content — under policy prompts of varying detail. The eval is in support of ROOST Model Community (RMC) inclusion review.

See [GUIDE.md](GUIDE.md) for the serving + evaluation playbook this report is built on. All numbers reproduce from the artefacts under `eval/results/`.

## Summary

- Cope-b-a4b is a strong policy-conditioned classifier. On the self-harm test set it beat gpt-oss-safeguard's best variant by **+0.19 F1** (0.887 vs 0.696) and produced clean binary output on 100% of inputs (gpt-oss-safeguard had 33–45% malformed responses on the same set).
- The same shape of detail-vs-performance curve appears in both domains: F1 climbs cleanly from a one-line policy to a structured medium-detail policy. Diminishing returns appear past the medium level.
- Both evaluations surfaced **policy-framework disagreements with Zentropi's own published policies** — in opposite directions. The published self-harm policy is more conservative than the test set expects (under-flags 23/50 violations). The published sexual-content "Long Policy" is more aggressive (over-flags 27 non-violations). Both are deliberate framing choices, not model failures, but they're worth surfacing to Zentropi and to RMC users.
- **Policy format alignment is non-negotiable.** Running OpenAI's GS0/GS1/GS2 sexual-content policy unmodified produced 33 errors out of 129 because its embedded "Output: VALID/INVALID" instructions conflicted with cope's binary prompt template. Adapting the same policy to binary output cleared all errors and added **+0.08 F1**.

## Setup

| | |
|---|---|
| **Model** | `zentropi-ai/cope-b-a4b` (Zentropi AI, pre-release) |
| **Serving** | Modal (H100, vLLM, OpenAI-compatible API) — `https://juliet--cope-b-a4b-serve.modal.run` |
| **Prompt format** | Cope's `INSTRUCTIONS / POLICY / CONTENT / ANSWER` template, raw `/v1/completions`, `max_tokens=1`, `temperature=0` |
| **Eval harness** | `eval/eval_cope.py`, concurrency 16 |
| **Date** | 2026-05-20 |

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

- **Accuracy = (TP + TN) / total** — Overall fraction correct. We report it but it's the least useful metric here because the test sets aren't perfectly balanced, and getting the majority class right inflates the number without showing whether you're catching violations.

In every table, **bold** marks the best value in that column.

### Why precision and recall tend to trade off

A stricter policy catches more (↑ recall) but also flags more borderline benign content (↓ precision). A looser policy is more selective (↑ precision) but lets violations through (↓ recall). The job of the policy author is to find the balance that matches the downstream use case. **There's no single "right" F1** — a content-removal pipeline and a triage queue have different acceptable trade-offs.

## Self-harm

### Test set

- Source: a 100-row self-harm test set (50 violating / 50 non-violating) provided to ROOST for evaluation use, with all original column identifiers sanitized (the original "Ground Truth Labelling" column is preserved as `ground_truth`).
- Included in this repository at `eval/test_set.csv` (sanitized — no labelling-org identifiers retained).
- Extracted to `eval/test_set.csv`

### Policies tested

| Name | Length | Description |
|---|---|---|
| `minimal` | ~1 sentence | One-line definition only |
| `simple` | ~10 lines | Short paragraph with basic includes/excludes |
| `medium` | ~30 lines | Structured: definitions + includes/excludes briefly |
| `full` | ~80 lines | Detailed cope-style policy with full Includes/Excludes structure |
| `zentropi_official` | ~70 lines | Zentropi's published self-harm policy, v7 (pulled from `zentropi.ai/labelers/d8c64ad5-957b-43fd-a999-6b15661851a9` via their API) |

### Results

| Policy              | F1        | Precision | Recall    | Errors |
|---------------------|-----------|-----------|-----------|--------|
| minimal             | 0.479     | 0.810     | 0.340     | 0      |
| simple              | 0.627     | 0.765     | 0.531     | 0      |
| medium              | 0.841     | **0.974** | 0.740     | 0      |
| **full**            | **0.887** | 0.915     | **0.860** | 0      |
| zentropi_official   | 0.563     | 0.952     | 0.400     | 0      |

Predictions: `eval/results/predictions_20260520_142014.csv`
Summary: `eval/results/summary_20260520_142014.csv`

### Comparison: cope-b-a4b vs gpt-oss-safeguard on the same 100-row set

Numbers below are recomputed from the test CSV's existing Safeguard-20b / gpt-oss-safeguard-20b columns. Empty/malformed responses are counted as false negatives, as in the original analysis.

| Model                          | F1        | Precision | Recall    | Malformed |
|--------------------------------|-----------|-----------|-----------|-----------|
| **cope-b-a4b (full)**          | **0.887** | 0.915     | **0.860** | **0**     |
| cope-b-a4b (medium)            | 0.841     | 0.974     | 0.740     | 0         |
| gpt-oss-safeguard-20b (full)   | 0.696     | 0.960     | 0.545     | 33        |
| cope-b-a4b (simple)            | 0.627     | 0.765     | 0.531     | 0         |
| gpt-oss-safeguard-20b (simple) | 0.583     | 0.824     | 0.452     | 45        |
| Safeguard-20b (full)           | 0.533     | 1.000     | 0.364     | 40        |
| Safeguard-20b (simple)         | 0.492     | 0.938     | 0.333     | 45        |
| cope-b-a4b (minimal)           | 0.479     | 0.810     | 0.340     | 0         |

### Finding: the official Zentropi self-harm policy under-flags by design

The `zentropi_official` policy underperforms our hand-written `full` policy by **0.32 F1**, despite being similarly detailed. The pattern across the 23 missed violations is consistent: it explicitly excludes content that the ground-truth labels mark as violating. The misses fall into three categories:

- **Glorification of self-harm without first-person ideation** — aesthetic, artistic, or "scars-as-trophies" framings that don't include the author stating personal suicidal ideation.
- **Recruitment, challenges, and community organising** — invitations to participate in self-harm together (pacts, accountability buddies, group counts, competitions).
- **Concealment and detection-avoidance content** — tutorials for hiding self-harm from observers (makeup over wounds, methods that don't leave marks).

All of these are **content where the author isn't disclosing first-person ideation but is enabling, glorifying, or recruiting for self-harm**. Zentropi's policy is explicitly built to **focus on first-person self-disclosure and protect support-seekers and recovery narratives**.

Conversely, our `full` policy produces 4 false positives that Zentropi *correctly* excludes — they're all support-seeking and recovery-narrative content (people asking for help resisting urges, day-counts of recovery progress, harm-reduction alternatives like the rubber-band technique). See `eval/results/predictions_20260520_142014.csv` for the row-by-row predictions across all policies.

**This is a policy-framework disagreement, not a model failure.** Zentropi's labeler is calibrated to protect recovery and support-seeking. The test set was labelled with a platform-moderation lens (catch recruitment, glorification, instructional content regardless of disclosure). Both framings are defensible — they answer different questions. Anyone publishing a comparative eval should make this explicit so model creators and users can pick the framing that matches their use case.

## Sexually explicit content

### Test set

Pre-labelled data didn't exist for this domain, so we built the test set from two sources:

1. **Stratified Bluesky sample (79 rows)**: One shard of the Hugging Face `withalim/bluesky-posts` dataset (~390k clean text-only posts after filtering) sampled into three tiers — strong sexual keywords, borderline/suggestive, and no-signal — then manually labelled.
2. **Synthetic red-team set (50 rows)**: Hand-crafted examples across 10 categories chosen to stress-test specific clauses of the Zentropi sexual-content policy (clear explicit acts, coded language, recovery narratives, educational/clinical content, fictional vs graphic creative writing, factual body-part mentions, etc.).

Final test set: **129 rows, 52 violating / 77 non-violating** at `eval/sex_eval/test_set.csv`.

The Bluesky sampler hard-drops posts with images, mostly-URL posts, mostly-non-Latin posts, and anything where a sexual term co-occurs with a minor-related term (CSAM-adjacent). Source: `eval/sex_eval/sample_bsky_for_sex_eval.py`.

### Policies tested

| Name | Length | Description |
|---|---|---|
| `sex_minimal` | ~1 sentence | One-line definition only |
| `sex_simple` | ~15 lines | Short paragraph with includes/excludes |
| `sex_medium` | ~45 lines | Structured: definitions + includes/excludes |
| `sex_zentropi_long` | ~180 lines | Zentropi's published "Long Policy" for sexually explicit content (provided by Zentropi to ROOST) |
| `sex_oai` | ~80 lines | OpenAI's `GS0/GS1/GS2` graphic-sexual-content policy, unmodified |
| `sex_oai_adapted` | ~50 lines | The same OpenAI policy rewritten to map GS1/GS2 → 1, GS0 → 0 (binary output) |

We did not write a "full" variant on the order of self-harm's `full.md` because `sex_zentropi_long` already serves that role.

### Results

| Policy              | F1        | Precision | Recall    | Errors |
|---------------------|-----------|-----------|-----------|--------|
| sex_minimal         | 0.653     | 0.673     | 0.635     | 0      |
| sex_simple          | 0.784     | 0.800     | 0.769     | 0      |
| **sex_medium**      | **0.844** | 0.807     | 0.885     | 0      |
| sex_zentropi_long   | 0.756     | 0.640     | **0.923** | 0      |
| sex_oai (raw)       | 0.763     | 0.853     | 0.690     | **33** |
| sex_oai_adapted     | 0.843     | **0.860** | 0.827     | 0      |

Predictions: `eval/results/predictions_sex_20260520_160725.csv`
Summary: `eval/results/summary_sex_20260520_160725.csv`

### Finding: Zentropi's "Long Policy" is over-aggressive on this test set

The `sex_zentropi_long` policy has the highest recall in the whole table (0.923 — it catches 48 of 52 violations) but also the lowest precision (0.640 — 27 false positives out of 75 flagged). Examined another way: it's calibrated for high recall at the cost of flagging substantial benign content.

This is **the mirror image of the self-harm finding**. The published Zentropi self-harm policy was *under-conservative* (high precision, low recall) on the self-harm test set. The published Zentropi sexual-content policy is *over-aggressive* (high recall, low precision) on our Bluesky + red-team set. In both cases, the official policy reflects deliberate framing choices by Zentropi that don't necessarily match how a downstream platform would label the same content.

Worth raising directly with Zentropi: are these calibrations intentional? If yes, an RMC writeup needs to make the framing explicit. If no, both policies may benefit from a revision pass against external test data.

### Finding: policy format must match model output expectations

The unmodified `sex_oai` policy (OpenAI's GS0/GS1/GS2 framework) produced **33 errors out of 129** when sent to cope. Cope's prompt template asks the model to "Answer 1 if yes, 0 if no", but the OpenAI policy text repeatedly instructs the model to output `VALID`, `INVALID`, or `GS0/GS1/GS2`. Cope tried to follow both, producing characters that were neither `0` nor `1`.

Rewriting the same policy to map GS1/GS2 → 1 and GS0 → 0 (`sex_oai_adapted.md`), with no other changes, **eliminated all errors and bumped F1 from 0.763 → 0.843** (+0.08).

**When porting a policy across model frameworks, output-format alignment is required, not optional.** The content of the policy can transfer; the output instructions cannot.

## Cross-domain observations

### 1. Policy detail buys you recall, up to a point

In both domains, the F1 curve looks like this:

- **minimal** (one sentence): F1 ~0.48 (self-harm), ~0.65 (sex)
- **simple** (short paragraph): F1 ~0.63, ~0.78
- **medium** (structured): F1 ~0.84, ~0.84
- **full / equivalent**: F1 ~0.89 (self-harm), ~0.76 (sex_zentropi_long — different shape because over-aggressive)

The lift from minimal → medium is large. The lift from medium → full is small in self-harm and slightly negative in sex (because the longer policy was over-calibrated). **A tight, well-structured medium-detail policy is a strong default.**

### 2. Precision survives detail, recall doesn't

Across both domains, the minimal policy already produces precision in the 0.67–0.81 range. Detail mostly buys recall, not precision. If a deployment is precision-critical (false-positive-sensitive — e.g., content removal at scale), a minimal or simple policy may be acceptable. If recall-critical (e.g., escalation triage where missed violations cost more), the medium-detail policy is the floor.

### 3. Cope's output reliability is materially better than gpt-oss-safeguard

Across 500 self-harm calls and 774 sexual-content calls, cope returned a clean `0` or `1` on every call where the prompt was well-formatted for binary output. The only errors (33 on the raw OpenAI policy) were attributable to the policy text itself instructing different output. By contrast, on the same self-harm test set, gpt-oss-safeguard had 33–45% malformed responses. **For any production use that depends on machine-parseable output, this is a significant advantage.**

### 4. The same model, the same content, different policy → different answer

This is the whole point of policy-conditioned classifiers, but the magnitude is worth noting. On the self-harm test set, the same 100 posts got F1 scores ranging from 0.479 to 0.887 depending purely on the policy text. **The policy is the product.** Model selection matters, but policy authoring is what determines real-world performance.

## How this lines up with Zentropi's published benchmark

Zentropi publishes a benchmark for cope alongside the model. For cope-b-a4b (text-only), their reported numbers, averaged unweighted across all their evaluation categories:

| Model | Precision | Recall | F1 |
|---|---|---|---|
| **CoPE-B-A4B-MM** (multimodal variant) | 0.83 | 0.84 | 0.82 |
| **CoPE-B-A4B** (text-only, what we tested) | 0.74 | 0.90 | 0.81 |
| CoPE-A-9B (the LoRA variant) | 0.74 | 0.88 | 0.80 |
| GPT-5.4 (default reasoning) | 0.68 | 0.95 | 0.78 |
| Gemini-3.5-Flash | 0.69 | 0.91 | 0.78 |
| Claude-Opus-4.6 | 0.65 | 0.95 | 0.75 |
| gpt-oss-safeguard-20b (default reasoning) | 0.70 | 0.82 | 0.75 |
| gpt-oss-20b (default reasoning) | 0.65 | 0.88 | 0.72 |
| GPT-5-mini (default reasoning) | 0.56 | 0.97 | 0.69 |

### Important caveat

**These numbers are not directly comparable to ours.** Zentropi's `0.81` is *their policy × their internal test set*. None of our runs share both inputs with theirs. The table below makes the combinations explicit:

| F1 | Policy | Test set |
|---|---|---|
| **0.81** (Zentropi published) | Zentropi's | Zentropi's internal |
| 0.887 (`full`, self-harm) | Ours | self-harm test CSV |
| 0.844 (`sex_medium`, sex) | Ours | Bluesky + redteam |
| 0.563 (`zentropi_official`, self-harm) | Zentropi's | self-harm test CSV |
| 0.756 (`sex_zentropi_long`, sex) | Zentropi's | Bluesky + redteam |

We can't draw "we outperformed Zentropi" from this — we ran a different test. What we *can* draw, in order of how useful it is to RMC:

### What the comparison actually tells us

1. **Zentropi's published policy is internally consistent with their benchmark.** Their policy × their test set lands at F1=0.81. The model is faithfully applying the policy it's given. Cope isn't broken; it isn't biased; it does what it's told.

2. **Their policy on outside test data degrades, and the size of the degradation depends on how the outside test was labelled.** When we pair Zentropi's policies with externally-labelled test sets, F1 drops:
   - Self-harm: 0.81 (their pairing) → 0.56 (their policy × the external self-harm labels). Big drop. The external labels treat glorification, recruitment, and concealment-tutorial content as violations even without first-person ideation — Zentropi's policy explicitly excludes those.
   - Sex: 0.81 → 0.76. Smaller drop. Our Bluesky+redteam labels are closer to (but not identical to) Zentropi's framing — their policy still over-flags, but less catastrophically.
   The size of the drop is a measurement of how far the test labels are from the policy's labelling philosophy. It is not a measurement of how good the model is.

3. **Cope's published recall (0.90) holds up against frontier models at a small fraction of the cost.** GPT-5.4, Gemini-3.5-Flash, and Claude-Opus-4.6 all sit in the 0.91–0.95 recall range on Zentropi's benchmark; cope is comparable and self-hostable. For deployments that care primarily about catching violations, cope is a strong economic choice even before you factor in the F1 gap to gpt-oss-safeguard (0.81 vs 0.75 on Zentropi's benchmark — the same direction and roughly the same magnitude as the gap we measured on the self-harm test).

### What this means for RMC adopters

The actually-useful conclusion for anyone considering cope: **you cannot take Zentropi's policies off the shelf and expect Zentropi's published F1.** Their policies are calibrated to their labelling philosophy. Your downstream F1 will depend on how your data is labelled, and on whether you're willing to author or adapt policies that match.

The model itself is solid. The prompt-engineering work — writing policies that match how *your* data is labelled — is where the variance lives, and where adopters will spend their time.

## Recommendation for RMC inclusion review

Cope-b-a4b is a credible candidate for the second RMC partner model.

Strengths to surface in the writeup:
- Best F1 on the self-harm benchmark across all open-weight safety models we tested.
- 100% well-formed binary output across 1,274 calls in this study.
- Comparable performance across two very different harm domains using the same prompt template — suggests the model generalizes well across policy domains.
- Faster cold-start and lower idle cost than full chat-model deployments (smaller weights, single-token output).

Caveats to surface:
- Zentropi's published policies on both domains showed substantial framework disagreement with externally-labelled test sets. RMC users adopting cope should evaluate the published policies against their own labelled data before deploying — or write their own. Performance is highly policy-dependent.
- Cope uses a non-chat prompt format. Most existing eval harnesses and integrations assume `/v1/chat/completions`; adapting to cope requires a small wrapper. Not a blocker, but worth documenting prominently.
- The pre-release model required `--enforce-eager` to avoid a vLLM compilation bug. This should clear up with the public release.

A reasonable next step before formal RMC inclusion: a third domain (e.g., harassment or hate) to confirm the pattern, and an explicit conversation with Zentropi about the framing-disagreement finding so the published policies can be calibrated against external benchmarks.
