#!/usr/bin/env python3
"""Fit the QLR projection on the log vectors and save it as `{out}_W.npy`, `{out}_b.npy`.

Same transform as the `efficient_pca` crate used by the submitted code: per-feature
standardization (z-score, std sanitized to 1 when ~0), then projection onto the top
`--dim` principal components (orthonormal, no whitening). Fitted on the first
`--n-fit` rows (100k, as before), or on the rows listed in `--rows` (first n-fit of them).

The folded form is y = W x + b with W = R^T diag(1/s), b = -W m.
"""
import argparse
import json
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--rows", help="npy int64 of rows to use")
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--n-fit", type=int, default=100_000)
    ap.add_argument("--out", required=True, help="output prefix")
    a = ap.parse_args()

    t0 = time.time()
    X = np.load(a.log, mmap_mode="r")
    rows = np.load(a.rows) if a.rows else np.arange(X.shape[0])
    fit_rows = np.sort(rows[: a.n_fit])
    Z = np.asarray(X[fit_rows], dtype=np.float64)
    m = Z.mean(axis=0)
    s = Z.std(axis=0, ddof=1)
    s = np.where(s > 1e-6, s, 1.0)
    Z = (Z - m) / s
    C = (Z.T @ Z) / (Z.shape[0] - 1)
    evals, evecs = np.linalg.eigh(C)
    R = evecs[:, ::-1][:, : a.dim]  # d x p, descending variance
    W = (R.T / s[None, :]).astype(np.float32)  # p x d
    b = (-(W.astype(np.float64) @ m)).astype(np.float32)
    np.save(f"{a.out}_W.npy", np.ascontiguousarray(W))
    np.save(f"{a.out}_b.npy", b)
    expl = float(evals[::-1][: a.dim].sum() / evals.sum())
    info = {"n_fit": int(len(fit_rows)), "in_dim": int(X.shape[1]), "out_dim": a.dim,
            "explained_variance": expl, "fit_s": time.time() - t0}
    with open(f"{a.out}_info.json", "w") as f:
        json.dump(info, f)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
