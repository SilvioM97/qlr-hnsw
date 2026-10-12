#!/usr/bin/env python3
"""The selected configurations of configs/*.csv for one dataset and method, in the form that
`qlr_search` takes: an ef list (--kind hnsw, printed) or a configurations file (--kind qlr or
catapult, written to --out; see `qlr_search --configs-file`).

  selected.py configs/e1_best.csv --dataset snowflake --method qlr --kind qlr --out qlr.csv
"""
import argparse

import pandas as pd

COLUMNS = {"qlr": ["k_prime", "th", "ef", "ef_min", "adaptive"],
           "catapult": ["lsh_bits", "bucket_size", "warmup", "entry", "ef"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--kind", required=True, choices=["hnsw", "qlr", "catapult"])
    ap.add_argument("--out")
    a = ap.parse_args()
    df = pd.read_csv(a.csv, dtype=str).fillna("")
    df = df[(df.dataset == a.dataset) & (df.method == a.method)]
    if a.kind == "hnsw":
        print(",".join(str(e) for e in sorted({int(x) for x in df.ef})))
        return
    if "adaptive" not in df.columns:
        df = df.assign(adaptive="true")
    df = df.assign(adaptive=df.adaptive.replace("", "true"))
    df[COLUMNS[a.kind]].drop_duplicates().to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
