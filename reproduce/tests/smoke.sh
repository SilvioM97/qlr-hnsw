#!/usr/bin/env bash
# Pipeline smoke test: the whole of run_all.sh on small synthetic data with tiny grids.
# Checks that every step runs and writes its results; the numbers are
# meaningless.  Usage: tests/smoke.sh <scratch-dir>
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
T=${1:?scratch directory}
export DATA=$T/data WORK=$T/work SMOKE=1 KMEANS_SAMPLE=40960 KMEANS_K=1024 E3_SIZE=40000
export PY=${PY:-python3} TIMING_CPU=${TIMING_CPU:-0} TIMING_NODE=${TIMING_NODE:-0}
[[ -d $DATA ]] || "$PY" "$HERE/tests/make_synthetic.py" --out "$DATA"
"$HERE/run_all.sh"
MODE=best "$HERE/run_all.sh"   # the selected configurations of configs/, on the same indexes
for f in e1_main e2_ablation e3_sources e4_window e6_logsize e8_cache e10_catapult; do
  [[ -s $WORK/results/$f.csv ]] || { echo "SMOKE TEST FAILED: missing results/$f.csv" >&2; exit 1; }
done
[[ -s $WORK/results/e9_leakage.json ]] || { echo "SMOKE TEST FAILED: missing results/e9_leakage.json" >&2; exit 1; }
for ds in minilm contriever snowflake mda msmarcov2 mdav2; do
  grep -q ",$ds," "$WORK/results/e1_main.csv" || { echo "SMOKE TEST FAILED: no E1 rows for $ds" >&2; exit 1; }
done
for ds in minilm contriever snowflake mda msmarcov2 mdav2; do
  grep -q ",$ds," "$WORK/results/best_e1.csv" || { echo "SMOKE TEST FAILED: no selected-configuration rows for $ds" >&2; exit 1; }
done
for ds in snowflake mda; do
  grep -q ",$ds," "$WORK/results/best_ablation.csv" || { echo "SMOKE TEST FAILED: no ablation rows for $ds" >&2; exit 1; }
done
echo "SMOKE TEST PASSED"
