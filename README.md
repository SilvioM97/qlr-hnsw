# QLR-HNSW

QLR-HNSW routes the ground-layer beam search of an HNSW index through a small index over a
query log. This repository contains the implementation (a fork of the kANNolo library) and the
scripts that run the experiments of the paper (`reproduce/`).

- `src/qlr.rs`: the router (`I_Q`, `EP`, routing score, adaptive `ef`).
- `src/catapult.rs`: our reimplementation of the CatapultDB baseline on HNSW.
- Binaries `qlr_build`, `qlr_search` and `kmeans_centroids`, next to kANNolo's `hnsw_build`.

## Installation

Linux x86_64 with `numactl` and Python >= 3.8.

1. Install Rust and the nightly toolchain (the toolchain is pinned in
   `rust-toolchain.toml`):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   rustup install nightly
   ```
2. Install the Python packages: `pip install numpy pandas pyarrow`.
3. Compile, optimized for the local CPU (the binaries are placed in `target/release/`):
   ```bash
   RUSTFLAGS="-C target-cpu=native" cargo build --release --features cli --bins
   ```

To check the installation, `reproduce/tests/smoke.sh <scratch-dir>` runs the whole pipeline on
small synthetic data with tiny grids (the numbers it produces are meaningless).

## Input data

All vectors are `.npy` matrices of `float32`, one row per item. Set `DATA` to a directory laid
out as follows:

```
$DATA/
  minilm/       docs.npy  queries.npy  train_queries.npy
  contriever/   docs.npy  queries.npy  train_queries.npy
  snowflake/    docs.npy  queries.npy  train_queries.npy  queries_ids.npy  train_queries_ids.npy
  msmarco_v2/   docs.npy  queries.npy                       (optional)
  mda/          queries.txt  embeddings/*.parquet
  msmarco_text/ queries.dev.tsv  queries.train.tsv           (only for the leakage check)
```

- **MS MARCO v1 passage** (8,841,823 passages), encoded with MiniLM (384 dimensions),
  Contriever (768, not normalized) and Snowflake (1024):
  - `docs.npy`: the passages, in collection order.
  - `queries.npy`: the 6,980 `dev.small` queries.
  - `train_queries.npy`: the 808,731 `train` queries, used as the query log.
  - `*_ids.npy`: the MS MARCO query ids of the rows (to join the query texts in the leakage check).
- **MS MARCO v2 passage** (138,364,198 passages), Snowflake only (optional; skipped if absent):
  `docs.npy` and `queries.npy`, the 3,903 `dev1` queries. Its query log is the v1 `train` set.
- **MDA**, one month of the MSN query log:
  - `queries.txt` is the raw log file `srfp20060501-20060531.queries.txt`.
  - `embeddings/` holds the Snowflake embeddings of its queries, as parquet files with columns
    `id` (the log's QueryID) and `embedding` (1,024 floats).
  - `reproduce/prepare_mda.py` derives the split: the test set is the last 10,000 queries of the
    log, and the query log holds the 1,483,054 distinct queries that occur at least twice before
    the test period. The MDA queries are searched against the MS MARCO v1 and v2 passages.

## Running the experiments

```bash
export DATA=/path/to/inputs WORK=/path/to/outputs PY=python3
export TIMING_CPU=2 TIMING_NODE=2   # core and NUMA node of the single-threaded timing runs
export V2_NODES=2,3                 # NUMA nodes over which the memory of the MS MARCO v2 runs is interleaved
reproduce/run_all.sh
```

`run_all.sh` runs the steps in order and skips every step whose output already exists, so it can
be restarted after an interruption. The steps can also be run one at a time. Datasets:
`minilm`, `contriever`, `snowflake`, `mda` (MDA queries on MS MARCO v1), `msmarcov2`,
`mdav2` (MDA queries on MS MARCO v2).

| Step | Command | Threads |
|---|---|---|
| MDA split | `prepare_mda.py` | 1 |
| HNSW indexes (m=32, `ef-construction`=500, f16) | `build.sh index <ds>` | all |
| Exact ground truth | `build.sh gt <ds>` | all |
| QLR routers | `build.sh router <ds>` | all |
| E3/E6 routers (131,072 k-means centroids on a 5.2M-passage sample) | `build.sh ablation-routers` | all |
| E1: HNSW and QLR-HNSW (Tables 1 and 2) | `run.sh e1 <ds>` | 1 |
| E2, E3: seeding only, entry sets (Table 3) | `run.sh e2/e3 <snowflake\|mda>` | 1 |
| E4: windows of the preceding queries | `run.sh e4` | 1 |
| E6 (log size), E8 (cache) | `run.sh e6`, `run.sh e8 <ds>` | 1 |
| E10: CatapultDB baseline (Table 3) | `run.sh e10 <snowflake\|mda>` | 1 |
| Leakage check | `e9_leakage.py` | all |

Rebuilt indexes differ slightly, because multi-threaded construction is not deterministic. The
timed runs dominate the total running time (a full grid on one core takes hours per dataset).
MS MARCO v2 needs about 370 GB of RAM to build its index (about 320 GB on disk), and its ground
truth is bound by reading the 567 GB `docs.npy`.

### Selected configurations

The grids are not needed to obtain the numbers of the tables. `reproduce/configs/` lists, for each
dataset, the fastest configuration found by the grids at each Accuracy@10 target:

- `e1_best.csv`: HNSW (`ef`) and QLR-HNSW (`k_prime`, `th`, `ef`, `ef_min`) for every dataset
  and for the targets used in the tables and in the text; `accuracy` is the Accuracy@10 of that
  configuration in our runs.
- `ablation_best.csv`: the seeding-only (fixed `ef`), entry-set and CatapultDB (`lsh_bits`,
  `bucket_size`, `warmup`, `entry`, `ef`) configurations on `snowflake` and `mda`, at 94, 96 and 98%.

`MODE=best reproduce/run_all.sh` builds the indexes and routers as above and then runs only these
configurations (`run.sh best <dataset>` for one dataset), writing `results/best_e1.csv` and
`results/best_ablation.csv`. The same configurations can be given to the binary directly:
`qlr_search --configs-file <csv>` runs exactly the rows of a CSV (`qlr` mode: columns
`k_prime,th,ef,ef_min,adaptive`; `catapult` mode: `lsh_bits,bucket_size,warmup,entry,ef`) instead
of the cross product of the `--*-list` options. Without `MODE=best`, `run_all.sh` runs the grids.

Timing rules:

- Timed runs are single-threaded and pinned with `numactl --physcpubind=$TIMING_CPU`. Their
  memory is bound to `$TIMING_NODE`, which therefore needs enough free memory for the index
  (about 20 GB for MS MARCO v1 with Snowflake); otherwise pages land on a remote node and
  latencies go up. The MS MARCO v2 runs need about 320 GB: set `V2_NODES` to interleave them over
  several nodes (check with `numastat -p <pid>` that nothing lands elsewhere).
- Never run a build step together with a timed run.
- Two timed runs can share the machine only if they are on different NUMA nodes.

## Outputs (`$WORK`)

- `results/*.csv`: one row per configuration (accuracy, mean latency, routing statistics),
  one file per experiment: `e1_main.csv` (HNSW and QLR-HNSW), `e2_ablation.csv`,
  `e3_sources.csv`, `e4_window.csv`, `e6_logsize.csv`, `e8_cache.csv`, `e10_catapult.csv`.
- `results/best_e1.csv`, `results/best_ablation.csv`: the selected configurations (`MODE=best`).
- `results/perquery/`: per-query statistics (recall, latency breakdown) of the E1 runs.
- `results/e9_leakage.json`: the leakage check.
- `indexes/`, `logs/`: the indexes and routers, and the logs of every step.

Latency is the mean single-thread latency per query. For QLR-HNSW it includes every step: PCA
projection, `I_Q` search, candidate scoring and beam search.
