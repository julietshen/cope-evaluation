"""Evaluate Sentinel on multi-turn grooming detection.

Stages:
  1. Build an index from UNannotated synthetic 1-line snippets.
  2. Fixed test: the expert-annotated 10-line conversations (aggregator
     comparison + top_k / min_score / index-size grid search).
  3. Composed long conversations (10-200 turns, controlled grooming density,
     risk tier, onset): conversation-level AUC per cell, line-level separation,
     whole-conversation vs sliding-window scoring, detection latency.
  4. Real predator-side chat logs (Perverted Justice) scored with the same
     windowed detector.

Writes CSVs + a JSON summary into results/.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.sentinel_local_index import SentinelLocalIndex
from sentinel.score_formulae import skewness, top_k_mean, max_score
from sentinel.simulation import (
    LabeledGroup,
    compare_aggregators,
    evaluate_groups,
    run_grid_search,
    score_groups,
)

import compose
import data

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42
TOP_K = 5
MIN_SCORE = 0.1
WINDOW = 10          # messages per sliding window (skewness needs >= 10)
STRIDE = 5

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- stage 1: index
log("Loading data...")
annotated = data.load_annotated_singles()
grooming_pools = annotated["pools"]

index_positives = data.load_single_line_snippets(
    "grooming", exclude_filenames=annotated["filenames"])
all_innocuous = data.load_single_line_snippets("innocuous")
index_negatives, composition_innocuous = data.split_pool(all_innocuous, seed=SEED)

log(f"Index seeds: {len(index_positives)} grooming (unannotated), "
    f"{len(index_negatives)} innocuous; composition pool: "
    f"{len(composition_innocuous)} innocuous + "
    f"{len(grooming_pools.high)}/{len(grooming_pools.med)}/{len(grooming_pools.low)} "
    f"grooming lines (high/med/low)")

log("Building index (encoding seeds)...")
index = SentinelLocalIndex.from_texts(
    positive_texts=index_positives,
    negative_texts=index_negatives,
    model_name="all-MiniLM-L6-v2",
    seed=SEED,
)
log("Index built.")

# ---------------------------- stage 2: 10-line convs (generated labels, noisy)
generated_convs = data.load_generated_10l(n_per_class=300, seed=SEED)
groups_10l = [LabeledGroup(name=n, label=l, observations=lines)
              for n, l, lines in generated_convs]
log(f"Scoring {len(groups_10l)} 10-line conversations "
    f"({sum(g.label for g in groups_10l)} generated-as-grooming)...")
scored_10l = score_groups(index, groups_10l, top_k=TOP_K)

agg_rows = compare_aggregators(scored_10l, min_score_to_consider=MIN_SCORE)
pd.DataFrame(agg_rows).to_csv(RESULTS / "gen10l_aggregators.csv", index=False)

log("Grid search over top_k / min_score / index size...")
grid_rows = run_grid_search(
    index, groups_10l,
    n_positive_values=[500, 2000, len(index_positives)],
    top_k_values=[3, 5, 10],
    min_score_values=[0.0, 0.1, 0.25],
    index_seed=SEED,
)
pd.DataFrame(grid_rows).to_csv(RESULTS / "gen10l_grid_search.csv", index=False)

# --------------------------------------------- stage 3: composed long convs
log("Composing long conversations...")
grid_convs = compose.build_grid(composition_innocuous, grooming_pools, seed=7)
late_convs = compose.build_late_onset_set(composition_innocuous, grooming_pools)
all_convs = grid_convs + late_convs
log(f"{len(grid_convs)} grid conversations + {len(late_convs)} late-onset")

log("Encoding unique pool lines once...")
unique_lines = sorted({line for conv in all_convs for line in conv.lines})
encoding_kwargs = dict(index.encoding_kwargs)
unique_embeddings = index.sentence_model.encode(unique_lines, **encoding_kwargs)
line_to_row = {line: i for i, line in enumerate(unique_lines)}
log(f"Encoded {len(unique_lines)} unique lines.")

conv_groups = [LabeledGroup(name=c.name, label=c.label, observations=c.lines)
               for c in all_convs]
conv_embeddings = {
    c.name: unique_embeddings[[line_to_row[l] for l in c.lines]] for c in all_convs
}
log("Scoring composed conversations...")
scored_convs = score_groups(index, conv_groups, top_k=TOP_K,
                            observation_embeddings=conv_embeddings)
scores_by_name = {s.name: s.observation_scores for s in scored_convs}
np.savez_compressed(RESULTS / "composed_observation_scores.npz", **scores_by_name)

# Line-level separation: do injected grooming lines score above background?
line_rows = []
for conv in grid_convs:
    if not conv.label:
        continue
    scores = scores_by_name[conv.name]
    positions = set(conv.grooming_positions)
    for i, s in enumerate(scores):
        line_rows.append({"tier": conv.tier, "is_grooming_line": i in positions,
                          "score": float(s)})
line_df = pd.DataFrame(line_rows)
line_df.to_csv(RESULTS / "composed_line_scores.csv", index=False)


def _auc(pos, neg):
    """Rank-based AUC for two 1-D score arrays."""
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


AGGREGATORS = {
    "skewness": skewness,
    "top_k_mean_3": lambda a: top_k_mean(a, k=3),
    "max_score": max_score,
}


def whole_conv_affinity(scores, aggregator):
    clipped = np.where(scores < MIN_SCORE, 0.0, scores)
    return float(aggregator(clipped)) if clipped.size else 0.0


def window_affinities(scores, aggregator, window=WINDOW, stride=STRIDE):
    """Affinity of each sliding window; a short conversation is one window."""
    clipped = np.where(scores < MIN_SCORE, 0.0, scores)
    if len(clipped) <= window:
        return np.array([float(aggregator(clipped))]), np.array([len(clipped)])
    starts = list(range(0, len(clipped) - window + 1, stride))
    if starts[-1] != len(clipped) - window:
        starts.append(len(clipped) - window)
    values = np.array([float(aggregator(clipped[s:s + window])) for s in starts])
    ends = np.array([s + window for s in starts])
    return values, ends


# Conversation-level metrics per grid cell, whole-conv and windowed-max.
cell_rows = []
lengths = sorted({c.length for c in grid_convs})
for agg_name, agg in AGGREGATORS.items():
    for length in lengths:
        negatives = [c for c in grid_convs if c.length == length and not c.label]
        neg_whole = np.array([whole_conv_affinity(scores_by_name[c.name], agg)
                              for c in negatives])
        neg_winmax = np.array([window_affinities(scores_by_name[c.name], agg)[0].max()
                               for c in negatives])
        for density in sorted({c.density for c in grid_convs if c.label}):
            for tier in ("high", "med", "low"):
                cell = [c for c in grid_convs
                        if (c.length, c.density, c.tier) == (length, density, tier)]
                if not cell:
                    continue
                pos_whole = np.array([whole_conv_affinity(scores_by_name[c.name], agg)
                                      for c in cell])
                pos_winmax = np.array(
                    [window_affinities(scores_by_name[c.name], agg)[0].max()
                     for c in cell])
                cell_rows.append({
                    "aggregator": agg_name, "length": length, "density": density,
                    "tier": tier, "n_pos": len(cell), "n_neg": len(negatives),
                    "auc_whole": _auc(pos_whole, neg_whole),
                    "auc_window_max": _auc(pos_winmax, neg_winmax),
                    "recall_at_fpr5_whole": float(
                        (pos_whole >= np.quantile(neg_whole, 0.95)).mean()),
                    "recall_at_fpr5_window": float(
                        (pos_winmax >= np.quantile(neg_winmax, 0.95)).mean()),
                })
cell_df = pd.DataFrame(cell_rows)
cell_df.to_csv(RESULTS / "composed_cell_metrics.csv", index=False)
log("Composed-grid metrics written.")

# Windowed detector threshold: 99th pct of window affinities on negatives.
thresholds = {}
for agg_name, agg in AGGREGATORS.items():
    neg_windows = np.concatenate([
        window_affinities(scores_by_name[c.name], agg)[0]
        for c in grid_convs if not c.label])
    thresholds[agg_name] = float(np.quantile(neg_windows, 0.99))
log(f"Window thresholds (99th pct of negative windows): {thresholds}")

# Detection latency on the late-onset set.
latency_rows = []
for agg_name, agg in AGGREGATORS.items():
    threshold = thresholds[agg_name]
    for conv in late_convs:
        values, ends = window_affinities(scores_by_name[conv.name], agg)
        hits = np.where(values >= threshold)[0]
        first_grooming = conv.grooming_positions[0]
        detected = len(hits) > 0
        latency_rows.append({
            "aggregator": agg_name, "conversation": conv.name,
            "detected": detected,
            "first_grooming_line": first_grooming,
            "n_grooming_lines": len(conv.grooming_positions),
            "detection_message": int(ends[hits[0]]) if detected else None,
            "latency_messages": (int(ends[hits[0]]) - first_grooming
                                 if detected else None),
        })
pd.DataFrame(latency_rows).to_csv(RESULTS / "late_onset_latency.csv", index=False)

# ------------------------------------------------- stage 4: real PJ chat logs
log("Scoring real Perverted Justice predator logs...")
pj_convs = data.load_pj_conversations(n=30, max_lines=150, seed=SEED)
pj_groups = [LabeledGroup(name=n, label=1, observations=lines)
             for n, lines in pj_convs]
scored_pj = score_groups(index, pj_groups, top_k=TOP_K)
pj_scores = {s.name: s.observation_scores for s in scored_pj}
np.savez_compressed(RESULTS / "pj_observation_scores.npz", **pj_scores)

pj_rows = []
for agg_name, agg in AGGREGATORS.items():
    threshold = thresholds[agg_name]
    for name, lines in pj_convs:
        values, ends = window_affinities(pj_scores[name], agg)
        hits = np.where(values >= threshold)[0]
        pj_rows.append({
            "aggregator": agg_name, "conversation": name, "n_lines": len(lines),
            "whole_conv_affinity": whole_conv_affinity(pj_scores[name], agg),
            "max_window_affinity": float(values.max()),
            "detected": len(hits) > 0,
            "detection_message": int(ends[hits[0]]) if len(hits) else None,
        })
pj_df = pd.DataFrame(pj_rows)
pj_df.to_csv(RESULTS / "pj_results.csv", index=False)

summary = {
    "index": {"n_positive": len(index_positives), "n_negative": len(index_negatives)},
    "window_thresholds_p99": thresholds,
    "pj_detection_rate": {
        agg: float(pj_df[pj_df.aggregator == agg].detected.mean())
        for agg in AGGREGATORS},
}
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
log("Done.")
