"""Dilution test: a FIXED number of grooming lines in growing conversations.

The main grid holds density constant, so longer conversations carry more
grooming lines and get easier. Here the count is held constant (2 or 5 lines)
while length grows 10 -> 400, which is the actual long-conversation failure
mode: the same brief grooming episode buried under more and more mundane chat.
Compares whole-conversation scoring against max-over-sliding-windows.
"""

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.sentinel_local_index import SentinelLocalIndex
from sentinel.score_formulae import skewness, top_k_mean, max_score
from sentinel.simulation import LabeledGroup, score_groups

import data
from compose import _sample

RESULTS = Path(__file__).resolve().parent / "results"
SEED = 42
TOP_K = 5
MIN_SCORE = 0.1
WINDOW, STRIDE = 10, 5

t0 = time.time()
annotated = data.load_annotated_singles()
pools = annotated["pools"]
index_positives = data.load_single_line_snippets(
    "grooming", exclude_filenames=annotated["filenames"])
all_innocuous = data.load_single_line_snippets("innocuous")
index_negatives, composition_innocuous = data.split_pool(all_innocuous, seed=SEED)

index = SentinelLocalIndex.from_texts(
    positive_texts=index_positives, negative_texts=index_negatives,
    model_name="all-MiniLM-L6-v2", seed=SEED)
print(f"[{time.time()-t0:.1f}s] index built")

rng = random.Random(19)
LENGTHS = (10, 25, 50, 100, 200, 400)
COUNTS = (2, 5)
convs = []  # (name, label, lines, length, n_grooming)
for length in LENGTHS:
    for i in range(40):
        convs.append((f"neg_L{length}_{i}", 0,
                      _sample(composition_innocuous, length, rng), length, 0))
    for count in COUNTS:
        if count >= length:
            continue
        for i in range(30):
            lines = _sample(composition_innocuous, length - count, rng)
            for pos, line in zip(sorted(rng.sample(range(length), count)),
                                 _sample(pools.high, count, rng)):
                lines.insert(pos, line)
            convs.append((f"pos_L{length}_g{count}_{i}", 1,
                          lines[:length], length, count))

unique = sorted({l for _, _, lines, _, _ in convs for l in lines})
emb = index.sentence_model.encode(unique, **dict(index.encoding_kwargs))
row = {l: i for i, l in enumerate(unique)}
groups = [LabeledGroup(name=n, label=lab, observations=lines)
          for n, lab, lines, _, _ in convs]
observation_embeddings = {n: emb[[row[l] for l in lines]]
                          for n, _, lines, _, _ in convs}
print(f"[{time.time()-t0:.1f}s] encoded {len(unique)} unique lines")
scored = score_groups(index, groups, top_k=TOP_K,
                      observation_embeddings=observation_embeddings)
scores = {s.name: s.observation_scores for s in scored}
print(f"[{time.time()-t0:.1f}s] scored {len(convs)} conversations")

AGGS = {"skewness": skewness, "top_k_mean_3": lambda a: top_k_mean(a, k=3),
        "max_score": max_score}


def whole(arr, agg):
    clipped = np.where(arr < MIN_SCORE, 0.0, arr)
    return float(agg(clipped))


def winmax(arr, agg):
    clipped = np.where(arr < MIN_SCORE, 0.0, arr)
    if len(clipped) <= WINDOW:
        return float(agg(clipped))
    starts = list(range(0, len(clipped) - WINDOW + 1, STRIDE))
    if starts[-1] != len(clipped) - WINDOW:
        starts.append(len(clipped) - WINDOW)
    return max(float(agg(clipped[s:s + WINDOW])) for s in starts)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return wins / (len(pos) * len(neg))


rows = []
for agg_name, agg in AGGS.items():
    for length in LENGTHS:
        negatives = [n for n, lab, _, L, _ in convs if L == length and lab == 0]
        for count in COUNTS:
            positives = [n for n, lab, _, L, g in convs
                         if (L, g, lab) == (length, count, 1)]
            if not positives:
                continue
            rows.append({
                "aggregator": agg_name, "length": length, "n_grooming": count,
                "auc_whole": auc([whole(scores[n], agg) for n in positives],
                                 [whole(scores[n], agg) for n in negatives]),
                "auc_window_max": auc([winmax(scores[n], agg) for n in positives],
                                      [winmax(scores[n], agg) for n in negatives]),
            })
df = pd.DataFrame(rows)
df.to_csv(RESULTS / "dilution_test.csv", index=False)
for agg_name in AGGS:
    sub = df[df.aggregator == agg_name]
    print("---", agg_name)
    print(sub.pivot_table(index="length", columns="n_grooming",
                          values=["auc_whole", "auc_window_max"]).round(3))
