#!/usr/bin/env python3
"""Seeded row subsets for E3 (entry-set source, 131,072 entries) and E6 (log size), as int64 npy
row files.

  snowflake_logrows_{10k,50k,131k,200k}.npy  random MS MARCO train queries (E3 q, E6)
  snowflake_docrows_131k.npy                 random documents (E3 d)
  mda_logrows_131k.npy                       random MDA log queries (E3 q on MDA)
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    data, der = os.environ["DATA"], os.path.join(os.environ["WORK"], "data")
    n_log = np.load(f"{data}/snowflake/train_queries.npy", mmap_mode="r").shape[0]
    n_docs = np.load(f"{data}/snowflake/docs.npy", mmap_mode="r").shape[0]
    n_mda = np.load(f"{der}/mda/train_queries.npy", mmap_mode="r").shape[0]
    e3 = int(os.environ.get("E3_SIZE", 131_072))  # reduced only by the smoke test (tests/smoke.sh)
    rng = np.random.default_rng(a.seed)
    pick = lambda n, size: np.sort(rng.choice(n, size=size, replace=False)).astype(np.int64)
    for size, tag in ((10_000, "10k"), (50_000, "50k"), (e3, "131k"), (200_000, "200k")):
        np.save(f"{a.out}/snowflake_logrows_{tag}.npy", pick(n_log, size))
    np.save(f"{a.out}/snowflake_docrows_131k.npy", pick(n_docs, e3))
    np.save(f"{a.out}/mda_logrows_131k.npy", pick(n_mda, e3))


if __name__ == "__main__":
    main()
