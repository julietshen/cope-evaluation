# Can Sentinel detect grooming in long multi-turn conversations?

**Evaluation report — August 2026.** Reproducible via `run_eval.py` and
`dilution_test.py` in this directory (~30 s each on an M-series Mac). All
result tables are in `results/`.

## Verdict in brief

**Yes for ranking/recall, with two significant caveats.** On expert-labeled
synthetic data and real Perverted Justice (PJ) predator logs, Sentinel
separates grooming from innocuous conversations with ROC AUC ≥ 0.95 in almost
every configuration tested, including a single grooming line hidden in a
200-message conversation, and flags real predator logs within the first 10–15
messages. The caveats:

1. **The default aggregator (skewness) is the wrong choice for grooming.** It
   scored at chance (AUC 0.49) on dense 10-turn grooming conversations and is
   provably blind to signal magnitude in sparse windows. Peak-focused
   aggregators (`top_k_mean`, `max_score`) dominate everywhere we measured.
2. **Real-world precision is unmeasurable with the data that exists today.**
   Every negative example available (synthetic innocuous chat) is stylistically
   unlike real casual chat — on real PJ logs, single-word greetings and
   casual filler score nearly as high as actual grooming lines. Recall claims
   survive this evaluation; precision claims cannot be made until the index and
   thresholds are calibrated against real platform traffic.

## Why this evaluation was needed

The shipped synthetic dataset has 1-line snippets (index seeds) and 10-line
conversations (short tests) — nothing resembling the hundreds-of-messages arc
of real grooming. Worse, the expert annotations for the 10-line conversations
are **unusable**: every filename in `annotations/multiple_mapping.csv` refers
to a Dec 9–10 generation batch that is not in the repo (shipped conversations
are Dec 18+), so none of the 200 conversation-level expert labels can be joined
to any text. (The 1-line annotations are fine — the CSV embeds the content.)

We therefore **composed** long multi-turn conversations from the expert-labeled
single lines: innocuous background with expert-confirmed grooming lines
(157 high / 93 med / 69 low risk) injected at controlled length (10–400
messages), density (5–20%), risk tier, and onset. Every composed conversation
has ground truth by construction, and the variables that matter — length,
signal density, escalation timing — are controlled independently. Leakage is
excluded by construction: the index is seeded only with *unannotated* snippets,
and the innocuous pool is split disjointly between index and eval.

**Setup.** Index: 4,501 unannotated grooming 1-liners (positives) + 2,500
synthetic innocuous 1-liners (negatives), `all-MiniLM-L6-v2`, `top_k=5`,
`min_score_to_consider=0.1`. Speaker tags stripped everywhere.

## Findings

### 1. The per-message signal is real

Across all composed conversations, injected grooming lines score a mean of
0.25–0.28 versus 0.002 for background lines. Notably, the expert's risk tiers
barely modulate the score (high 0.278, med 0.251, low 0.261) — Sentinel sees
"grooming-flavored language" but not graded severity, so downstream triage
cannot use the score as a severity signal.

### 2. Multi-turn detection is strong on controlled data

Conversation-level AUC on the composed grid (`top_k_mean`, whole-conversation
scoring; 20 positives/cell vs 40 negatives/length):

| Length | 5% density | 10% | 20% |
|---|---|---|---|
| 10 | 0.93 | 0.92 | 0.95 |
| 25 | 0.93 | 1.00 | 1.00 |
| 50 | 0.98 | 1.00 | 1.00 |
| 100 | 1.00 | 1.00 | 1.00 |
| 200 | 1.00 | 1.00 | 1.00 |

At fixed density, longer conversations are *easier* (more absolute signal).
Low-risk-tier lines detect about as well as high — consistent with finding 1.

### 3. The true long-conversation stressor is dilution — manageable with the right configuration

Holding the grooming episode *fixed* (2 or 5 lines) while the conversation
grows is the realistic hard case. AUC with 2 grooming lines
(`results/dilution_test.csv`):

| Length | skewness, whole conv | top_k_mean, whole | max_score, whole | top_k_mean, windowed max |
|---|---|---|---|---|
| 10 | 1.00 | 1.00 | 1.00 | 1.00 |
| 100 | 0.97 | 0.96 | 0.98 | 0.98 |
| 200 | 0.98 | 0.98 | 0.99 | 0.98 |
| 400 | **0.84** | 0.92 | 0.96 | 0.95 |

Two grooming messages in 400 remain detectable at AUC 0.95–0.96 with
peak-focused aggregators; skewness degrades badly (0.84 whole-conversation,
0.71 windowed). With 5 grooming lines every configuration stays ≥ 0.99 except
windowed skewness. **Detection latency** is excellent when scoring is windowed:
in late-onset conversations (grooming begins ~message 60 of 100), every
aggregator detects within a median of 4 messages after the first grooming line.

### 4. The default skewness aggregator should not be used for this task

Two distinct failure modes, both reproducible:

- **Dense grooming collapses it.** On the shipped 10-line conversations
  (generated labels), where most lines are grooming-flavored, skewness scored
  AUC **0.49** — chance — while `max_score` reached 0.97 and a tuned
  `top_k_mean` 0.99. Skewness assumes rare signal among mostly-normal
  messages; when the majority of a window is grooming, the median rises and
  the "asymmetry" vanishes. Sustained, blatant grooming is exactly the case a
  detector must not miss.
- **It is magnitude-blind in sparse windows.** For a window whose scores are
  zero except one positive value v, skewness = (v/10)/(0.3v) = **1/3
  regardless of v**. One borderline line scoring 0.11 and one flagrant line
  scoring 2.0 produce identical window scores — which is why the skewness
  false-positive threshold in our run landed at exactly 0.333.

Recommendation: `top_k_mean` (k≈3) over a sliding window of ~10–50 messages,
with `max_score` as a simpler near-equivalent.

### 5. Real predator logs: 100% flagged, quickly — but with an asterisk

Against 30 randomly sampled real PJ predator-side logs (first 150 messages
each), using a threshold set at the 99th percentile of synthetic-negative
window scores: **30/30 detected, median detection by message 10–15**, i.e.
within the first one or two windows. A median 23% of each predator's messages
score above the per-message threshold.

The asterisk, seen by inspecting top-scoring lines: genuine grooming probes
score highest (physical-contact questions, checks on whether a parent is
nearby or can enter the child's room), but generic one-word greetings and
casual filler score nearly as high (0.27–0.30). The synthetic innocuous
negatives are hobby-focused, well-punctuated prose, so the index has partly
learned *casual chat register* as a grooming feature. Real innocuous teen chat
— full of exactly that casual filler — would fire far more often than our
synthetic negatives suggest. Two further reasons to treat PJ numbers as an upper bound:
PJ conversations were included in the Gemma3 prompt that generated the index
seeds (style contamination), and PJ decoy logs are unusually grooming-dense
because the decoys never discourage the predator.

## Limitations

- **No real innocuous chat anywhere in the pipeline** — the central gap.
  Precision/FPR numbers here describe separation from *synthetic* innocuous
  text only. Deployment decisions need thresholds calibrated on real platform
  traffic (see IMPLEMENTATION_GUIDE.md §4).
- Composed conversations splice independent lines; they have no discourse
  coherence. This should not flatter per-message embedding scores, but it
  cannot capture cross-message escalation cues either — nor can Sentinel,
  which scores messages independently.
- English only; `all-MiniLM-L6-v2`; single synthetic data source (Gemma3);
  expert labels from one annotator.
- Grooming lines reused across composed conversations (pools are small);
  conversations are not fully independent samples.

## Recommendations

1. **Deploy config**: sliding window of 10–50 messages per sender (per
   counterparty where possible), `top_k_mean(k=3)`, `min_score_to_consider≈0.1`,
   threshold from your own traffic's negative percentiles, human review behind
   every flag. Details in [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md).
2. **Do not ship the skewness default for grooming** without understanding §4;
   at minimum document the dense-conversation failure mode upstream.
3. **Data asks for the dataset authors**: republish the Dec 9–10 10-line batch
   (or re-annotate the shipped batch) so conversation-level expert labels are
   usable; add real or realistic *casual-register* innocuous chat; consider
   multi-annotator labels.
4. **Before any production claim**: rebuild the index with real platform
   negatives and re-run `run_eval.py` — the harness is data-agnostic.
