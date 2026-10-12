#!/usr/bin/env bash
# Build steps (all threads). Never run them while a timing run is in progress.
#
#   build.sh gt <dataset>                     exact top-10 ground truth of the test queries
#   build.sh index <minilm|contriever|snowflake|msmarcov2>   HNSW, m=32, ef-construction=500, f16
#   build.sh router <dataset>                 QLR router over the dataset's query log
#   build.sh ablation-routers                 E3 (131k entry sets) and E6 (log size) routers
# (the datasets are listed in config.sh; mda and mdav2 use the indexes of snowflake and msmarcov2)
#
# The index build time goes to $LOGS/e5_build_id_<dataset>.log (used by the cost analysis).
source "$(dirname "$0")/config.sh"
STEP=$1; shift

router() {  # dataset [tag log-file [rows.npy]]
  dataset_vars "$1"
  local tag=${2:-} logf=${3:-$LOG} rows=${4:-}
  local out=$ROUTER${tag:+_$tag} pca=$IDX/${NAME}${tag:+_$tag}_pca$PCA_DIM
  # PCA to d/4, fitted on (the first 100k rows of) the log; I_Q m=32, efc=1000; EP: k_ep=10
  # documents per log query, found with ef=200.
  "$PY" "$HERE/fit_pca.py" --log "$logf" ${rows:+--rows "$rows"} --dim "$PCA_DIM" --out "$pca"
  "$BIN/qlr_build" --doc-index "$DOC" --log-file "$logf" ${rows:+--log-rows "$rows"} \
    --projection-prefix "$pca" --k-ep 10 --query-index-m 32 --query-index-ef-construction 1000 \
    --training-search-ef 200 -o "$out" --times-json "$out.json" > "$LOGS/build_$(basename "$out").log" 2>&1
  cat "$out.json"; echo
}

case $STEP in
  gt)
    dataset_vars "$1"
    "$PY" "$HERE/groundtruth.py" --docs "$DOCS" --queries "$QUERIES" --out "$GT" ;;
  index)
    dataset_vars "$1"
    /usr/bin/time -v "$BIN/hnsw_build" --data-file "$DOCS" --output-file "$DOC.tmp" \
      --dataset-type dense --value-type f16 --encoder plain --graph-type fixed-degree \
      --m 32 --ef-construction 500 --distance dotproduct > "$LOGS/e5_build_id_$NAME.log" 2>&1
    mv "$DOC.tmp" "$DOC" ;;   # only a complete index gets its final name
  router)
    router "$1" ;;
  ablation-routers)
    "$PY" "$HERE/subsets.py" --out "$IDX"
    dataset_vars snowflake
    # E3 (c): 131,072 spherical k-means centroids trained on a 5.2M-document sample (40 points
    # per centroid).
    [[ -f $IDX/snowflake_kmeans131k.npy ]] || "$BIN/kmeans_centroids" --data-file "$DOCS" \
      --sample "${KMEANS_SAMPLE:-5242880}" -k "${KMEANS_K:-131072}" -o "$IDX/snowflake_kmeans131k.npy" > "$LOGS/kmeans131k.log" 2>&1
    router snowflake q131k "$LOG" "$IDX/snowflake_logrows_131k.npy"
    router snowflake doc131k "$DOCS" "$IDX/snowflake_docrows_131k.npy"
    router snowflake kmeans131k "$IDX/snowflake_kmeans131k.npy"
    for n in 10k 50k 200k; do router snowflake "q$n" "$LOG" "$IDX/snowflake_logrows_$n.npy"; done
    # MDA shares the documents, so the doc/centroid routers above serve it too.
    dataset_vars mda
    router mda q131k "$LOG" "$IDX/mda_logrows_131k.npy" ;;
  *) echo "unknown step $STEP" >&2; exit 1 ;;
esac
