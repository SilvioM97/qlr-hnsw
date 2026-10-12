#!/usr/bin/env python3
"""E9 — leakage check between MS MARCO dev.small (test) and train (Q_L).

Reports, for the dev.small queries:
  1. fraction with an exact duplicate in train after string normalization
     (lowercase, punctuation -> space, collapsed whitespace);
  2. fraction with cosine >= 0.95 to some train query (Snowflake embeddings, exact scan).
Also writes the row indices of the "leaky" dev queries (either criterion).

Text: MS MARCO queries.tar.gz (queries.dev.tsv, queries.train.tsv), joined on the qid arrays
$DATA/snowflake/{queries_ids,train_queries_ids}.npy. The join is validated by checking that identical
normalized texts have near-identical embeddings.
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default=os.path.join(os.environ.get("DATA", "."), "snowflake"))
    ap.add_argument("--text-dir", default=os.path.join(os.environ.get("DATA", "."), "msmarco_text"))
    ap.add_argument("--cos-th", type=float, default=0.95)
    ap.add_argument("--out", default=os.path.join(os.environ.get("WORK", "."), "results", "e9_leakage"))
    a = ap.parse_args()

    dev_ids = [int(x) for x in np.load(f"{a.emb_dir}/queries_ids.npy")]
    tr_ids = [int(x) for x in np.load(f"{a.emb_dir}/train_queries_ids.npy")]
    rd = lambda f: pd.read_csv(f, sep="\t", header=None, names=["qid", "text"], quoting=3,
                               dtype={"qid": int, "text": str})
    dmap = dict(rd(f"{a.text_dir}/queries.dev.tsv").values)
    tmap = dict(rd(f"{a.text_dir}/queries.train.tsv").values)
    assert all(i in dmap for i in dev_ids) and all(i in tmap for i in tr_ids)

    tnorm = {}
    for r, i in enumerate(tr_ids):
        tnorm.setdefault(norm(tmap[i]), r)
    exact = np.array([norm(dmap[i]) in tnorm for i in dev_ids])

    Qd = np.load(f"{a.emb_dir}/queries.npy").astype(np.float32)
    Qt = np.load(f"{a.emb_dir}/train_queries.npy", mmap_mode="r")
    Qd /= np.linalg.norm(Qd, axis=1, keepdims=True)
    best = np.full(len(Qd), -np.inf, dtype=np.float32)
    for s in range(0, Qt.shape[0], 100_000):
        B = np.array(Qt[s:s + 100_000], dtype=np.float32)
        B /= np.linalg.norm(B, axis=1, keepdims=True)
        best = np.maximum(best, (Qd @ B.T).max(axis=1))
    near = best >= a.cos_th

    # Join validation: identical normalized text => cosine ~ 1.
    pair_cos = [float(Qd[r] @ (Qt[tnorm[norm(dmap[i])]] / np.linalg.norm(Qt[tnorm[norm(dmap[i])]])))
                for r, i in enumerate(dev_ids) if exact[r]]
    leaky = np.where(exact | near)[0].astype(np.int64)
    res = {
        "n_dev": len(dev_ids), "n_train": len(tr_ids),
        "exact_dup": int(exact.sum()), "exact_dup_frac": float(exact.mean()),
        "cos_th": a.cos_th, "near_dup": int(near.sum()), "near_dup_frac": float(near.mean()),
        "either": int(len(leaky)), "either_frac": float(len(leaky) / len(dev_ids)),
        "join_check_median_cos_identical_text": float(np.median(pair_cos)) if pair_cos else None,
        "best_cos_percentiles": {p: float(np.percentile(best, p)) for p in (50, 90, 99)},
    }
    np.save(f"{a.out}_leaky_rows.npy", leaky)
    np.save(f"{a.out}_clean_rows.npy", np.setdiff1d(np.arange(len(dev_ids)), leaky).astype(np.int64))
    with open(f"{a.out}.json", "w") as f:
        json.dump(res, f, indent=1)
    pd.DataFrame([{k: v for k, v in res.items() if k != "best_cos_percentiles"}]).to_csv(f"{a.out}.csv", index=False)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
