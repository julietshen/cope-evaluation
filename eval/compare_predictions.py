"""Compare two predictions_*.csv runs on a shared policy column — the
format-sensitivity / A-B check the eval leaned on twice (raw-vs-harmony for
safeguard, criterion-in-Query for Shieldstral).

Reports per-policy P/R/F1 for each run and the number of prediction flips.

Usage:
    python compare_predictions.py A.csv B.csv --policy medium
    python compare_predictions.py A.csv B.csv --policy medium --label-a generic --label-b criterion
"""
from __future__ import annotations

import argparse
import csv

csv.field_size_limit(10 ** 9)


def prf(preds, truth):
    tp = sum(p == "1" and t == "1" for p, t in zip(preds, truth))
    fp = sum(p == "1" and t == "0" for p, t in zip(preds, truth))
    fn = sum(p == "0" and t == "1" for p, t in zip(preds, truth))
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F


def load(path, policy):
    rows = list(csv.DictReader(open(path)))
    col = f"{policy}_pred"
    if col not in rows[0]:
        raise SystemExit(f"{path} has no column {col}; has: "
                         f"{[c for c in rows[0] if c.endswith('_pred')]}")
    return {r["id"]: (r[col], r["ground_truth"]) for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    A = load(args.a, args.policy)
    B = load(args.b, args.policy)
    ids = [i for i in A if i in B]
    truth = [A[i][1] for i in ids]
    pa = [A[i][0] for i in ids]
    pb = [B[i][0] for i in ids]

    Pa, Ra, Fa = prf(pa, truth)
    Pb, Rb, Fb = prf(pb, truth)
    flips = sum(1 for x, y in zip(pa, pb) if x != y)

    print(f"policy: {args.policy}   shared rows: {len(ids)}")
    print(f"  {args.label_a:12s} P={Pa:.3f} R={Ra:.3f} F1={Fa:.3f}")
    print(f"  {args.label_b:12s} P={Pb:.3f} R={Rb:.3f} F1={Fb:.3f}")
    print(f"  prediction flips: {flips}/{len(ids)}  "
          f"(dF1={Fb-Fa:+.3f} dR={Rb-Ra:+.3f} dP={Pb-Pa:+.3f})")


if __name__ == "__main__":
    main()
