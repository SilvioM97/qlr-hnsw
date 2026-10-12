#!/usr/bin/env bash
# Timing runs (single-threaded, pinned with numactl; see config.sh). Never run two of them on
# the same core, nor while a build step is in progress. Rows are appended to $RES/<exp>.csv.
#
#   run.sh e1 <dataset>   HNSW and QLR-HNSW, full grid (Tables 1 and 2; the seeding-only and
#                         ablation rows of Table 3 use the same HNSW and QLR-HNSW rows)
#   run.sh e2 <snowflake|mda>   seeding only, fixed ef (Table 3)
#   run.sh e3 <snowflake|mda>   131k entry sets: queries, documents, k-means centroids (Table 3)
#   run.sh e4             MDA windows of the w = 1 ... 10k preceding queries (text)
#   run.sh e6             MS MARCO / Snowflake, |Q_L| = 10k, 50k, 200k
#   run.sh e8 <snowflake|mda>   approximate cache (top-10 of the closest past query)
#   run.sh e10 <snowflake|mda>  CatapultDB-style baseline, reimplemented on HNSW (Table 3)
#   run.sh best <dataset>       only the selected configurations of configs/ (no grid): E1 for every
#                               dataset, and the seeding-only, entry-set and CatapultDB rows for
#                               snowflake and mda. Rows go to $RES/best_e1.csv, $RES/best_ablation.csv.
source "$(dirname "$0")/config.sh"
EXP=$1; dataset_vars "${2:-mda}"
COMMON=(--doc-index "$DOC" --queries "$QUERIES" --gt "$GT" --dataset "$NAME")
qs() {
  [[ $NAME == msmarcov2 || $NAME == mdav2 ]] && drop_cache "$DOCS"
  "${NUMA[@]}" "$BIN/qlr_search" "$@"
}

case $EXP in
  e1)
    # QLR runs, each "k' list | th list | ef_min list | ef list". The ef floors above 10 use a
    # coarser ef grid on MDA; on MDA, the 99% point needs ef beyond 600; HNSW reaches 99% there.
    HNSW_EF=$EF_LIST
    case $NAME in
      mda) RUNS=("$KP_E1|$TH_LIST|10|$EF_LIST" "$KP_E1|$TH_LIST|15,20,40|$EF_MDA_COARSE"
                 "$KP_ABL|0.3,0.4,0.5,0.6,0.7|20,40|$EF_EXT") ;;
      msmarcov2) HNSW_EF="$EF_LIST,$EF_EXT"; RUNS=("$KP_E1|$TH_LIST|$EF_MIN|$EF_LIST") ;;
      mdav2) HNSW_EF="$EF_600,$EF_EXT"; RUNS=("$KP_ABL|$TH_LIST|$EF_MIN|$EF_LIST") ;;
      *) RUNS=("$KP_E1|$TH_LIST|$EF_MIN|$EF_LIST") ;;
    esac
    qs --mode hnsw "${COMMON[@]}" --ef-list "$HNSW_EF" --experiment e1 --method hnsw \
      --out-csv "$RES/e1_main.csv" --per-query-dir "$RES/perquery/e1" > "$LOGS/e1_${NAME}_hnsw.log" 2>&1
    for r in "${RUNS[@]}"; do
      IFS='|' read -r kp th em efs <<< "$r"
      qs --mode qlr "${QLR_FLAGS[@]}" "${COMMON[@]}" --router "$ROUTER" --ef-list "$efs" \
        --kprime-list "$kp" --th-list "$th" --adaptive true --ef-min-list "$em" \
        --experiment e1 --method qlr --out-csv "$RES/e1_main.csv" \
        --per-query-dir "$RES/perquery/e1" >> "$LOGS/e1_${NAME}_qlr.log" 2>&1
    done ;;
  e2)
    qs --mode qlr "${QLR_FLAGS[@]}" "${COMMON[@]}" --router "$ROUTER" --ef-list "$EF_LIST" \
      --kprime-list "$KP_E1" --th-list "$TH_LIST" --adaptive false --experiment e2 \
      --method qlr_seed_only --out-csv "$RES/e2_ablation.csv" > "$LOGS/e2_$NAME.log" 2>&1 ;;
  e3)
    # Same mechanism for every source: s_max of the dataset's full query log (a router over
    # documents would otherwise have s_max = 1, since each document is its own top-1).
    Q3=$("$PY" -c "import json;print(json.load(open('$ROUTER.json'))['q3'])")
    SF=$IDX/snowflake_router_pca256_kep10
    for src in q131k doc131k kmeans131k; do
      if [[ $src == q131k ]]; then R=${ROUTER}_q131k; else R=${SF}_$src; fi
      qs --mode qlr "${QLR_FLAGS[@]}" "${COMMON[@]}" --router "$R" --q3-override "$Q3" \
        --ef-list "$EF_LIST" --kprime-list "$KP_ABL" --th-list "$TH_LIST" --experiment e3 \
        --method "qlr_$src" --out-csv "$RES/e3_sources.csv" > "$LOGS/e3_${NAME}_$src.log" 2>&1
    done ;;
  e4)
    # For each test query, in log order, the router is the w queries that immediately precede
    # it in the raw log (w = 1 reuses the previous query), with their EP computed on I_D. The
    # window is scanned exhaustively; the scan is timed separately and left out of the latency.
    S=$DER/mda
    win() {  # w-list k'-list
      qs --mode window "${QLR_FLAGS[@]}" "${COMMON[@]}" --router "$ROUTER" --stream "$S/stream.npy" \
        --test-stream-pos "$S/test_stream_pos.npy" --query-order "$S/test_chrono_order.npy" \
        --w-list "$1" --kprime-list "$2" --th-list "$TH_LIST" --ef-list "$EF_MDA_COARSE" \
        --experiment e4 --method qlr_window --out-csv "$RES/e4_window.csv" >> "$LOGS/e4_window.log" 2>&1
    }
    win 1,10 10           # k' > w is the same as k' = w
    win 100,1000,3000,10000 "$KP_ABL" ;;
  e6)
    dataset_vars snowflake
    for n in 10k 50k 200k; do
      qs --mode qlr "${QLR_FLAGS[@]}" --doc-index "$DOC" --queries "$QUERIES" --gt "$GT" \
        --dataset snowflake --router "${ROUTER}_q$n" --ef-list "$EF_LIST" --kprime-list "$KP_ABL" \
        --th-list "$TH_LIST" --experiment e6 --method "qlr_log$n" --out-csv "$RES/e6_logsize.csv" \
        > "$LOGS/e6_$n.log" 2>&1
    done ;;
  e8)
    # When s >= th, return EP(q_1), the top-10 of the closest log query, without searching;
    # otherwise fall back to HNSW with the given ef.
    qs --mode qlr "${QLR_FLAGS[@]}" --action cache "${COMMON[@]}" --router "$ROUTER" \
      --kprime-list 10 --th-list "$TH_STD,0.8,0.9,0.95" --ef-list 10,50,100,200 --experiment e8 \
      --method cache_ep_q1 --out-csv "$RES/e8_cache.csv" > "$LOGS/e8_$NAME.log" 2>&1 ;;
  e10)
    # CatapultDB (arXiv:2603.02164, Algorithm 2) on the same HNSW: L hyperplanes, buckets of b
    # catapults (top-1 results of earlier queries, LRU), starting points = bucket + default
    # entry, fixed ef, online updates in query order; cold start and warm start from Q_L.
    # Entry `descent`: bucket + HNSW upper-level descent for every query; `fallback`: bucket
    # only, with the descent only for empty buckets.
    ORDER=(); [[ $NAME == mda ]] && ORDER=(--query-order "$DER/mda/test_chrono_order.npy")
    qs --mode catapult "${COMMON[@]}" "${ORDER[@]}" --router "$ROUTER" --log-file "$LOG" \
      --lsh-bits-list 6,8,10,12 --bucket-size-list 10,20,40,80 --catapult-warmup-list none,log \
      --catapult-entry-list descent,fallback \
      --ef-list "$EF_LIST" --experiment e10 --method catapult --out-csv "$RES/e10_catapult.csv" \
      > "$LOGS/e10_$NAME.log" 2>&1 ;;
  best)
    CFG=$HERE/configs
    sel() { "$PY" "$HERE/selected.py" "$@"; }
    T=$LOGS/best_$NAME   # configuration files of this run
    qs --mode hnsw "${COMMON[@]}" --ef-list "$(sel "$CFG/e1_best.csv" --dataset "$NAME" --method hnsw --kind hnsw)" \
      --experiment best_e1 --method hnsw --out-csv "$RES/best_e1.csv" > "$LOGS/best_e1_${NAME}_hnsw.log" 2>&1
    sel "$CFG/e1_best.csv" --dataset "$NAME" --method qlr --kind qlr --out "$T.e1.cfg"
    qs --mode qlr "${QLR_FLAGS[@]}" "${COMMON[@]}" --router "$ROUTER" --configs-file "$T.e1.cfg" \
      --experiment best_e1 --method qlr --out-csv "$RES/best_e1.csv" > "$LOGS/best_e1_${NAME}_qlr.log" 2>&1
    if [[ $NAME == snowflake || $NAME == mda ]]; then
      Q3=$("$PY" -c "import json;print(json.load(open('$ROUTER.json'))['q3'])")
      SF=$IDX/snowflake_router_pca256_kep10
      ORDER=(); [[ $NAME == mda ]] && ORDER=(--query-order "$DER/mda/test_chrono_order.npy")
      for m in qlr_seed_only qlr_q131k qlr_doc131k qlr_kmeans131k; do
        case $m in
          qlr_seed_only) R=$ROUTER ;;
          qlr_q131k) R=${ROUTER}_q131k ;;
          *) R=${SF}_${m#qlr_} ;;
        esac
        sel "$CFG/ablation_best.csv" --dataset "$NAME" --method "$m" --kind qlr --out "$T.$m.cfg"
        qs --mode qlr "${QLR_FLAGS[@]}" "${COMMON[@]}" --router "$R" --q3-override "$Q3" \
          --configs-file "$T.$m.cfg" --experiment best_ablation --method "$m" \
          --out-csv "$RES/best_ablation.csv" > "$LOGS/best_ablation_${NAME}_$m.log" 2>&1
      done
      sel "$CFG/ablation_best.csv" --dataset "$NAME" --method catapult --kind catapult --out "$T.catapult.cfg"
      qs --mode catapult "${COMMON[@]}" "${ORDER[@]}" --router "$ROUTER" --log-file "$LOG" \
        --configs-file "$T.catapult.cfg" --experiment best_ablation --method catapult \
        --out-csv "$RES/best_ablation.csv" > "$LOGS/best_ablation_${NAME}_catapult.log" 2>&1
    fi ;;
  *) echo "unknown experiment $EXP" >&2; exit 1 ;;
esac
