#!/usr/bin/env python3
"""Small synthetic inputs with the layout of $DATA (README.md), for the pipeline smoke test.

Clustered vectors (so that HNSW reaches high accuracy), queries near documents, a query log
with repeated queries, and an MDA-like raw log with parquet embeddings. Sizes are the minimum
the pipeline needs (E6 uses 200k log queries; E3 uses E3_SIZE documents and MDA log queries,
131,072 by default, 40,000 in the smoke test).
"""
import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def clustered(rng, centers, n, noise, normalize=True):
    x = centers[rng.integers(len(centers), size=n)] + noise * rng.standard_normal((n, centers.shape[1]))
    x = x.astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True) if normalize else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-docs", type=int, default=100_000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for name, dim, norm, n_train in (("minilm", 384, True, 20_000), ("contriever", 768, False, 20_000),
                                     ("snowflake", 1024, True, 210_000)):
        d = f"{a.out}/{name}"; os.makedirs(d, exist_ok=True)
        centers = rng.standard_normal((256, dim)).astype(np.float32)
        docs = clustered(rng, centers, a.n_docs, 0.6, norm)
        if not norm:
            docs *= rng.uniform(1, 3, size=(len(docs), 1)).astype(np.float32)
        np.save(f"{d}/docs.npy", docs)
        np.save(f"{d}/queries.npy", clustered(rng, centers, 500, 0.6, norm))
        train = clustered(rng, centers, n_train, 0.6, norm)
        np.save(f"{d}/train_queries.npy", train)
        if name == "snowflake":
            np.save(f"{d}/queries_ids.npy", np.arange(500, dtype=np.int64))
            np.save(f"{d}/train_queries_ids.npy", np.arange(1000, 1000 + n_train, dtype=np.int64))
            t = f"{a.out}/msmarco_text"; os.makedirs(t, exist_ok=True)
            with open(f"{t}/queries.dev.tsv", "w") as f:
                for i in range(500):
                    f.write(f"{i}\tquery {i % 50}\n")       # some dev texts also occur in train
            with open(f"{t}/queries.train.tsv", "w") as f:
                for i in range(n_train):
                    f.write(f"{1000 + i}\tquery {i}\n")
            sf_centers = centers

    # MS MARCO v2: more documents from the same clusters, and its own dev1 queries.
    d = f"{a.out}/msmarco_v2"; os.makedirs(d, exist_ok=True)
    np.save(f"{d}/docs.npy", clustered(rng, sf_centers, 2 * a.n_docs // 3, 0.6))
    np.save(f"{d}/queries.npy", clustered(rng, sf_centers, 300, 0.6))

    # MDA: 70k distinct texts seen twice, 20k seen once, then 10k test rows (half repeats).
    d = f"{a.out}/mda"; os.makedirs(f"{d}/embeddings", exist_ok=True)
    n_rep, n_once, n_test = 70_000, 20_000, 10_000
    texts = [f"t{i}" for i in range(n_rep)] * 2 + [f"u{i}" for i in range(n_once)]
    rng.shuffle(texts)
    texts += [f"t{rng.integers(n_rep)}" if i % 2 else f"v{i}" for i in range(n_test)]
    tvec = {}
    with open(f"{d}/queries.txt", "w") as f:
        f.write("Time\tQuery\tQueryID\tSessionID\tResultCount\n")
        for i, t in enumerate(texts):
            f.write(f"2006-05-01 00:00:{i:08d}\t{t}\tq{i}\ts{i // 3}\t10\n")
    ids, embs = [], []
    for i, t in enumerate(texts):
        if t not in tvec:
            tvec[t] = clustered(rng, sf_centers, 1, 0.6)[0]
        ids.append(f"q{i}"); embs.append(tvec[t])
    E = np.stack(embs)
    for p, s in enumerate(range(0, len(ids), 50_000)):
        tab = pa.table({"id": ids[s:s + 50_000],
                        "embedding": pa.array(list(E[s:s + 50_000]), type=pa.list_(pa.float32()))})
        pq.write_table(tab, f"{d}/embeddings/part{p:03d}.parquet")
    print("synthetic data written to", a.out)


if __name__ == "__main__":
    main()
