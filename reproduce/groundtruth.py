#!/usr/bin/env python3
"""Exact top-k by inner product (exhaustive scan in f32), the reference for Accuracy@10.

Writes `query_row \t doc_row \t rank \t score` (rank from 1), the format read by qlr_search.
  groundtruth.py --docs docs.npy --queries queries.npy --out groundtruth.tsv [--k 10]
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--doc-block", type=int, default=1_000_000)
    ap.add_argument("--query-block", type=int, default=1_000)
    a = ap.parse_args()
    D = np.load(a.docs, mmap_mode="r")
    Q = np.ascontiguousarray(np.load(a.queries), dtype=np.float32)
    nq, k = len(Q), a.k
    best_s = np.full((nq, k), -np.inf, dtype=np.float32)
    best_i = np.zeros((nq, k), dtype=np.int64)
    for d0 in range(0, len(D), a.doc_block):
        B = np.ascontiguousarray(D[d0:d0 + a.doc_block], dtype=np.float32)
        for q0 in range(0, nq, a.query_block):
            S = Q[q0:q0 + a.query_block] @ B.T
            top = np.argpartition(-S, k - 1, axis=1)[:, :k]
            s = np.take_along_axis(S, top, axis=1)
            cs = np.concatenate([best_s[q0:q0 + len(S)], s], axis=1)
            ci = np.concatenate([best_i[q0:q0 + len(S)], top + d0], axis=1)
            keep = np.argpartition(-cs, k - 1, axis=1)[:, :k]
            best_s[q0:q0 + len(S)] = np.take_along_axis(cs, keep, axis=1)
            best_i[q0:q0 + len(S)] = np.take_along_axis(ci, keep, axis=1)
        print(f"[gt] {min(d0 + a.doc_block, len(D))}/{len(D)} documents", flush=True)
    order = np.argsort(-best_s, axis=1, kind="stable")
    best_s = np.take_along_axis(best_s, order, axis=1)
    best_i = np.take_along_axis(best_i, order, axis=1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for q in range(nq):
            for r in range(k):
                f.write(f"{q}\t{best_i[q, r]}\t{r + 1}\t{best_s[q, r]:.7f}\n")


if __name__ == "__main__":
    main()
