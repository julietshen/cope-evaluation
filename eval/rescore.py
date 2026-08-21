"""Re-score existing predictions against a different label set.

eval.py scores every policy column against the ground_truth of the test set it
was run with. When the same content has multiple valid labelings (e.g. scam-only
vs scam+spam), the model predictions are identical — only the scoring differs.
This re-scores a predictions CSV against an alternate labels CSV (joined on id),
so a single inference pass covers multiple scopes.

Usage:
    python rescore.py 'results/predictions_cope_b_cope_b_scamall_*.csv' \
        --labels scam_eval/test_set_spam.csv [--policies scam_spam_inclusive scam_simple]
"""
from __future__ import annotations
import argparse, csv, glob
from pathlib import Path


def metrics(pred, truth):
    tp=sum(p=="1" and t=="1" for p,t in zip(pred,truth))
    fp=sum(p=="1" and t=="0" for p,t in zip(pred,truth))
    fn=sum(p=="0" and t=="1" for p,t in zip(pred,truth))
    tn=sum(p=="0" and t=="0" for p,t in zip(pred,truth))
    err=sum(p=="" for p in pred)
    prec=tp/(tp+fp) if tp+fp else 0.0
    rec=tp/(tp+fn) if tp+fn else 0.0
    f1=2*prec*rec/(prec+rec) if prec+rec else 0.0
    acc=(tp+tn)/max(len(pred)-err,1)
    return tp,fp,fn,tn,err,prec,rec,f1,acc


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--policies", nargs="*", help="policy stems to score (default: all *_pred columns)")
    a=ap.parse_args()
    path=Path(sorted(glob.glob(a.predictions))[-1])
    rows=list(csv.DictReader(open(path)))
    labels={r["id"]:r["ground_truth"] for r in csv.DictReader(open(a.labels))}
    pols=a.policies or [c[:-5] for c in rows[0] if c.endswith("_pred")]
    truth=[labels[r["id"]] for r in rows]
    print(f"file: {path.name}   labels: {Path(a.labels).name}   n={len(rows)} (pos {truth.count('1')})")
    print(f"{'policy':28s}{'tp':>4}{'fp':>4}{'fn':>4}{'tn':>4}{'err':>4}{'prec':>7}{'rec':>7}{'f1':>7}{'acc':>7}")
    for p in pols:
        col=f"{p}_pred"
        if col not in rows[0]: continue
        pred=[r[col] for r in rows]
        tp,fp,fn,tn,err,pr,rc,f1,ac=metrics(pred,truth)
        print(f"{p:28s}{tp:>4}{fp:>4}{fn:>4}{tn:>4}{err:>4}{pr:>7.3f}{rc:>7.3f}{f1:>7.3f}{ac:>7.3f}")


if __name__=="__main__":
    main()
