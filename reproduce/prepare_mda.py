#!/usr/bin/env python3
"""MDA split (Sect. 3): from the raw MSN query log and the Snowflake embeddings of its queries.

Inputs ($DATA/mda):
  queries.txt          the raw log (srfp20060501-20060531.queries.txt): a header line, then
                       Time, Query, QueryID, SessionID, ResultCount, tab-separated,
                       in chronological order
  embeddings/*.parquet columns `id` (QueryID) and `embedding` (list<float>, 1024-d)

Split:
  test  = the last 10,000 rows of the log
  Q_L   = the distinct (lowercased) query texts occurring at least twice in the rows before
          the test period, each represented by the embedding of its first occurrence
  stream (E4) = the last --w-max rows before the test period, plus the test rows

Outputs ($WORK/data/mda): queries.npy, query_ids.npy, train_queries.npy (chronological order);
stream.npy, test_stream_pos.npy, test_chrono_order.npy, stream_meta.csv; split.json.
Rows whose QueryID has no embedding are dropped from Q_L and reported (one in the paper's log).
"""
import argparse
import glob
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

N_TEST = 10_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.environ.get("DATA", "."), "mda"))
    ap.add_argument("--out", default=os.path.join(os.environ.get("WORK", "."), "data", "mda"))
    ap.add_argument("--w-max", type=int, default=10_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # Same parsing as the original split: skip the header, strip, split on tabs, drop
    # malformed lines.
    rows = []
    with open(f"{a.data}/queries.txt") as f:
        next(f)
        for line in f:
            s = line.strip().split("\t")
            if len(s) == 5:
                rows.append(s)
    log = pd.DataFrame(rows, columns=["time", "text", "qid", "session", "result_count"])
    del rows
    n = len(log)
    t0 = n - N_TEST
    train, test = log.iloc[:t0], log.iloc[t0:]

    lower = train.text.str.lower()
    counts = Counter(lower)
    first = ~lower.duplicated(keep="first")
    ql = train[first & lower.map(counts).ge(2)]

    stream = log.iloc[t0 - a.w_max:].reset_index(drop=True)
    want = set(test.qid) | set(ql.qid) | set(stream.qid)
    emb = {}
    for fn in sorted(glob.glob(f"{a.data}/embeddings/*.parquet")):
        t = pq.read_table(fn, columns=["id", "embedding"])
        ids = t.column("id").to_pylist()
        keep = [i for i, q in enumerate(ids) if q in want]
        if not keep:
            continue
        col = t.column("embedding").combine_chunks()
        dim = len(col[0])
        vals = np.asarray(col.flatten(), dtype=np.float32).reshape(-1, dim)
        for i in keep:
            emb[ids[i]] = vals[i]

    def stack(qids, what, drop_missing=False):
        miss = [q for q in qids if q not in emb]
        if miss and not drop_missing:
            raise SystemExit(f"{len(miss)} {what} rows have no embedding")
        return np.stack([emb[q] for q in qids if q in emb]), len(miss)

    Q, _ = stack(list(test.qid), "test")
    L, l_miss = stack(list(ql.qid), "Q_L", drop_missing=True)
    S, _ = stack(list(stream.qid), "stream")
    np.save(f"{a.out}/queries.npy", Q)
    np.save(f"{a.out}/query_ids.npy", test.qid.to_numpy())
    np.save(f"{a.out}/train_queries.npy", L)
    np.save(f"{a.out}/stream.npy", S)
    pos = np.arange(a.w_max, a.w_max + N_TEST, dtype=np.int64)  # test rows are the stream tail
    np.save(f"{a.out}/test_stream_pos.npy", pos)
    np.save(f"{a.out}/test_chrono_order.npy", np.arange(N_TEST, dtype=np.int64))
    stream.assign(is_test=stream.index >= a.w_max).to_csv(f"{a.out}/stream_meta.csv", index=False)
    info = {"log_rows": n, "test": len(Q), "q_l": len(L), "q_l_without_embedding": l_miss,
            "stream_rows": len(S), "w_max": a.w_max}
    json.dump(info, open(f"{a.out}/split.json", "w"))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
