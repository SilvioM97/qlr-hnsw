#!/usr/bin/env bash
# Whole pipeline, in order: data preparation, builds (all threads), timing runs (one core),
# leakage check. Each step is skipped if its output already exists, so the
# script can be re-run after an interruption. See README.md for the inputs.
#   MODE=grid (default)  the experiments, over their full grids
#   MODE=best            only the selected configurations of configs/ (run.sh best), no grids
source "$(dirname "$0")/config.sh"
R=$HERE

step() { echo "[$(date '+%F %T')] $*"; }

# 1. MDA split and E4 stream.
[[ -f $DER/mda/split.json ]] || { step "prepare MDA"; "$PY" "$R/prepare_mda.py"; }

# MS MARCO v2 (138M passages; index ~320 GB on disk, ~370 GB of RAM to build) is run only if
# its embeddings are present: $DATA/msmarco_v2/{docs.npy,queries.npy}.
V1=(minilm contriever snowflake mda); V2=()
if [[ -f $DATA/msmarco_v2/docs.npy ]]; then V2=(msmarcov2 mdav2); else step "no $DATA/msmarco_v2: skipping MS MARCO v2"; fi

# 2. Builds: HNSW indexes, ground truth, routers (all threads).
INDEXES=(minilm contriever snowflake); (( ${#V2[@]} )) && INDEXES+=(msmarcov2)
for ds in "${INDEXES[@]}"; do
  dataset_vars $ds
  [[ -f $DOC ]] || { step "build index $ds"; "$R/build.sh" index $ds; }
done
for ds in "${V1[@]}" "${V2[@]}"; do
  dataset_vars $ds
  [[ -f $GT ]] || { step "ground truth $ds"; "$R/build.sh" gt $ds; }
  [[ -f $ROUTER ]] || { step "build router $ds"; "$R/build.sh" router $ds; }
done
[[ -f $IDX/mda_router_pca256_kep10_q131k ]] || { step "build ablation routers"; "$R/build.sh" ablation-routers; }

# 3. Timing runs (single-threaded, one at a time).
done_mark() { [[ -f $LOGS/$1.done ]]; }
timed() {  # name, then run.sh arguments
  local name=$1; shift
  done_mark "$name" && return
  step "run $*"; "$R/run.sh" "$@"; touch "$LOGS/$name.done"
}
if [[ ${MODE:-grid} == best ]]; then
  for ds in "${V1[@]}" "${V2[@]}"; do timed "best_$ds" best $ds; done
else
  for ds in "${V1[@]}" "${V2[@]}"; do timed "e1_$ds" e1 $ds; done
  for ds in snowflake mda; do timed "e2_$ds" e2 $ds; timed "e3_$ds" e3 $ds; timed "e8_$ds" e8 $ds; timed "e10_$ds" e10 $ds; done
  timed e4 e4
  timed e6 e6
fi

# 4. Leakage check (MS MARCO dev vs train).
[[ -f $RES/e9_leakage.json ]] || { step "E9 leakage"; "$PY" "$R/e9_leakage.py"; }
step "done: results in $RES"
