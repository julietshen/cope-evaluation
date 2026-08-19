# Implementing Sentinel for grooming detection: a practical guide

This guide is for trust & safety engineers evaluating or deploying
[Sentinel](https://github.com/Roblox/Sentinel) to detect grooming in multi-turn chat. It
distills what the library's own docs assume you already know: what Sentinel
actually is (and is not), how to feed it conversation data, which knobs matter,
and how to wire it into a production moderation pipeline. Numbers referenced
here come from the companion evaluation ([REPORT.md](REPORT.md), reproducible
via `run_eval.py`).

## 1. What Sentinel is — and what it is not

Sentinel is a **high-recall candidate generator**, not a classifier you act on
directly. It has three moving parts:

1. **An index** of embedded example texts: *positives* (grooming-like single
   messages) and *negatives* (ordinary chat). No model training is involved;
   "improving the model" means curating these examples.
2. **A per-message contrastive score**: each incoming message is embedded and
   compared to its nearest neighbors on both sides of the index. Messages more
   similar to grooming examples than to ordinary chat get a positive score.
3. **An aggregator** that collapses the scores of *one sender's recent
   messages* into a single affinity number (default: skewness of the score
   distribution).

Two design consequences matter for deployment:

- **The unit of detection is a sender, not a message or a conversation.**
  Sentinel answers: "do this person's recent messages contain a pattern of
  grooming-like language?" Score each participant's messages separately;
  don't concatenate both sides of a conversation.
- **Output is a ranking signal.** The affinity score orders senders for human
  review or downstream (more expensive, higher-precision) analysis. It is not
  calibrated evidence, and thresholds only mean anything relative to a
  baseline you measure on your own traffic.

## 2. Building the index

### Seed data

- **Positives**: short, single-message examples of grooming language across
  stages (trust-building, isolation, boundary-testing, escalation). The
  synthetic 1-line snippets in
  the Sentinel repo's `examples/synthetic-grooming-conversations/` are exactly this shape. If you
  have expert-labeled data (e.g. ICMEC-style risk tiers), spend it wisely:
  use *unreviewed* bulk data for the index and reserve expert labels for
  evaluation — you need trustworthy ground truth for tuning far more than the
  index needs perfect seeds.
- **Negatives**: ordinary chat from *your* platform if at all possible. The
  negative side defines "normal"; synthetic or off-platform negatives make
  everything that looks like your real traffic slightly abnormal, which
  inflates false positives.

```python
from sentinel.sentinel_local_index import SentinelLocalIndex

index = SentinelLocalIndex.from_texts(
    positive_texts=grooming_lines,      # single messages, not whole conversations
    negative_texts=ordinary_lines,
    model_name="all-MiniLM-L6-v2",
    seed=42,                            # reproducible negative downsampling
)
index.save(path="s3://…/sentinel-grooming-v1",
           encoder_model_name_or_path="all-MiniLM-L6-v2")
```

Version the saved index directory like a model artifact: the index *is* the
model. Rebuild and re-evaluate when you add examples; `subsample()` +
`sentinel.simulation.run_grid_search` let you sweep index size and
positive:negative ratio cheaply before committing.

### Index hygiene

- Strip usernames/speaker tags from seed lines; you want the index to match
  language, not names.
- Deduplicate near-identical seeds — a cluster of duplicates acts as one
  overweighted example.
- Keep a held-out expert-labeled set that never touches the index. Any line
  used as a seed is contaminated for evaluation purposes.

## 3. Feeding it conversations: the buffering layer

Sentinel's API takes a list of recent messages; production systems have event
streams. The bridge is a per-sender rolling buffer:

```
message event ──► append to buffer[sender_id]   (ring buffer, size W)
                        │
                        ▼  every message (or every k messages)
              affinity = index.calculate_rare_class_affinity(buffer[sender_id])
                        │
                        ▼
              affinity ≥ threshold ──► enqueue sender for review
```

Practical parameters (validated in the evaluation):

- **Window size W ≈ 10–50 messages per sender.** The skewness aggregator needs
  at least 10 scores to return a non-zero value (`min_size_of_scores=10`);
  below that a sender is invisible, so also alert on high *individual* message
  scores for very low-volume senders if that gap worries you.
- **Do not score an entire long history at once with skewness.** Grooming
  signal concentrated in one phase of a months-long chat is diluted by
  thousands of mundane messages. A sliding window (or max-over-windows, if you
  score retrospectively) preserves detection as conversations grow.
- **Score per sender per counterparty pair if your platform allows it** —
  grooming language is targeted at one victim; pooling one sender's messages
  across many recipients dilutes exactly like a long history does.
- **Cadence**: scoring every message is cheapest correctness-wise (one
  embedding per message, amortized); if cost-bound, score every k=5 messages.
  Cache message embeddings — an embedding never changes, and re-encoding the
  window on every event is the main avoidable cost.

## 4. Choosing the aggregator and thresholds

- Use `sentinel.simulation` with labeled groups from your own platform to pick
  the aggregator; don't take the default on faith. In our evaluation the
  default skewness failed two ways that matter for grooming (REPORT.md §4): it
  collapses to chance when *most* of a window is grooming-flavored (the median
  rises, the asymmetry vanishes), and in sparse windows it is magnitude-blind
  (one line scoring 0.11 and one scoring 2.0 give the same window score).
  Peak-focused aggregators — `top_k_mean(k=3)`, or `max_score` — dominated in
  every condition we measured and are the recommended starting point.
- Calibrate the alert threshold as a percentile of affinity scores on a large
  sample of *known-benign* traffic (e.g. the 99th or 99.9th percentile),
  then verify recall on your labeled positives. Recalibrate whenever you
  rebuild the index or change the embedding model — raw scores are not
  comparable across index versions.
- Budget review capacity first, then set the threshold to fill it: Sentinel is
  a ranking system, and "top N senders per day" is often a more honest
  operating point than a fixed score cutoff.

## 5. What happens after a hit

Sentinel deliberately trades precision for recall. Design the funnel
accordingly:

1. **Sentinel flags a sender** (cheap, runs on everything).
2. **A higher-precision stage** examines the flagged sender's full recent
   conversation — an LLM classifier, a purpose-built grooming model, or
   direct human review depending on volume and policy.
3. **Human decision** for enforcement and (where legally required) reporting,
   e.g. NCMEC in the US.

The per-message `explanations` in `RareClassAffinityResult` (top matching
index examples per message) are genuinely useful in the review UI: they show
*which* messages drove the flag and *which known pattern* they matched.

## 6. Operational cautions

- **Domain shift is the main silent failure.** An index seeded with synthetic
  or off-platform text underperforms on your platform's slang, code words, and
  age-typical writing. Plan a feedback loop: confirmed true positives from
  review become index positives; persistent false-positive patterns become
  negatives.
- **Adversarial drift**: groomers adapt phrasing; embedding similarity is more
  robust than keyword lists but not immune. Track score distributions over
  time; a slow decline in flagged-sender confirm rate may mean the index is
  aging.
- **Evasion via language/register**: evaluate on the languages and age groups
  your platform actually has. `all-MiniLM-L6-v2` is English-centric; swap in a
  multilingual sentence-transformer and rebuild if you need coverage.
- **Privacy & legal**: the buffer is a store of minors' message content;
  scope retention to the window size, encrypt at rest, and gate the review UI.
  Anonymization of index seeds matters too — an index built from real victim
  conversations embeds that text; synthetic or paraphrased seeds avoid
  shipping victim data inside a model artifact.
- **Skewness edge cases**: fewer than 10 messages → score is exactly 0 (not
  "low risk" — *no signal*). Distinguish the two in dashboards.

## 7. Reproducing the evaluation

```bash
git clone https://github.com/Roblox/Sentinel        # for the library + synthetic dataset
python -m venv .venv && .venv/bin/pip install -e 'Sentinel[sbert]' pandas

# Point the harness at the datasets (defaults assume ~/ROOST/…):
export SYNTH_GROOMING_DATA=Sentinel/examples/synthetic-grooming-conversations
export PJ_DATASET=/path/to/PervertedJusticeDataset  # optional; only stage 4 needs it

.venv/bin/python run_eval.py       # main evaluation (~30 s); writes results/
.venv/bin/python dilution_test.py  # fixed-episode long-conversation stressor
```

See [REPORT.md](REPORT.md) for the findings and their limitations.
