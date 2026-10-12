# Source this file. Everything is configured through environment variables:
#
#   DATA   input vectors (read-only), laid out as described in README.md
#   WORK   output root: derived data, indexes, results and logs
#   PY     python with numpy, pandas and pyarrow (default: python3)
#   TIMING_CPU / TIMING_NODE   core and NUMA node of the single-threaded timing runs.
#          Pick a node with enough free memory for the index (~20 GB for MS MARCO/Snowflake)
#          so that no page is allocated on a remote node.
#   V2_NODES   optional, e.g. "2,3": NUMA nodes over which the memory of the MS MARCO v2 runs is
#          interleaved, when its index (~320 GB) does not fit in the node TIMING_NODE.
set -euo pipefail
: "${DATA:?set DATA to the input directory (see README.md)}"
: "${WORK:?set WORK to the output directory}"
PY=${PY:-python3}
TIMING_CPU=${TIMING_CPU:-0}
TIMING_NODE=${TIMING_NODE:-0}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BIN=${BIN:-$HERE/../target/release}
DER=$WORK/data; IDX=$WORK/indexes; RES=$WORK/results; LOGS=$WORK/logs
mkdir -p "$DER" "$IDX" "$RES" "$LOGS"

# Single-threaded timing: one core, memory of its own node only (see dataset_vars for MS MARCO v2).
NUMA=(numactl --physcpubind="$TIMING_CPU" --membind="$TIMING_NODE")

# Drops files from the page cache (a timed run reads its index once; cached pages of the large
# input files would otherwise take the memory of the node the index should be placed on).
drop_cache() {
  local f
  for f in "$@"; do
    "$PY" -c "import os,sys; fd=os.open(sys.argv[1], os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)" "$f" || true
  done
}

# Grids (Sect. 3 of the paper). Routing scores of Contriever are not cosines (unnormalized
# embeddings), hence its extended th grid.
EF_200="10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200"
EF_400="$EF_200,220,240,260,280,300,320,340,360,380,400"
EF_500="$EF_400,420,440,460,480,500"
EF_600="$EF_500,520,540,560,580,600"
EF_EXT="650,700,800,900,1000"   # beyond 600: MS MARCO v2 (HNSW) and the 99% point on MDA (QLR)
EF_MDA_COARSE="10,20,40,60,80,100,120,140,160,180,200,240,280,320,360,400,440,480,520,560,600"
TH_STD="0.1,0.2,0.3,0.4,0.5,0.6,0.7"
TH_V2_MDA="0.1,0.2,0.3,0.4,0.5"  # MDA log on MS MARCO v2
KP_E1="5,10,20"            # E1, E2
KP_ABL="10,20"             # E3, E4 (w >= 100), E6: same grid on both sides of each comparison
EF_MIN="10,15,20,40"       # E1 (adaptive beam width floor)
# SMOKE=1: tiny grids for the pipeline test on synthetic data (tests/smoke.sh), not for results.
if [[ ${SMOKE:-0} == 1 ]]; then
  EF_200="10,20,40,80"; EF_400=$EF_200; EF_500=$EF_200; EF_600="10,20,40,80,160"
  EF_EXT="320,640"; EF_MDA_COARSE=$EF_600
  TH_STD="0.2,0.5"; TH_V2_MDA="0.2,0.5"; KP_E1="5,10"; EF_MIN="10,20"
fi
# QLR reuses the routing scores of the candidates as seed distances (bit-identical results).
QLR_FLAGS=(--reuse-seed-scores)

# Datasets: minilm, contriever, snowflake (MS MARCO v1, dev.small queries, train queries as the
# log), mda (MDA queries and log, MS MARCO v1 passages), msmarcov2 (MS MARCO v2, dev1 queries,
# the v1 train queries as the log), mdav2 (MDA queries and log, MS MARCO v2 passages).
dataset_vars() {
  NUMA=(numactl --physcpubind="$TIMING_CPU" --membind="$TIMING_NODE")
  case "$1" in
    minilm)     DIM=384;  SRC=$DATA/minilm;     EF_LIST=$EF_200; TH_LIST=$TH_STD ;;
    contriever) DIM=768;  SRC=$DATA/contriever; EF_LIST=$EF_500; TH_LIST=$TH_STD ;;
    snowflake)  DIM=1024; SRC=$DATA/snowflake;  EF_LIST=$EF_200; TH_LIST=$TH_STD ;;
    mda)        DIM=1024; SRC=$DATA/snowflake;  EF_LIST=$EF_600; TH_LIST=$TH_STD ;;
    msmarcov2)  DIM=1024; SRC=$DATA/msmarco_v2; EF_LIST=$EF_600; TH_LIST=$TH_STD ;;
    mdav2)      DIM=1024; SRC=$DATA/msmarco_v2; EF_LIST=$EF_MDA_COARSE; TH_LIST=$TH_V2_MDA ;;
    *) echo "unknown dataset $1" >&2; return 1 ;;
  esac
  NAME=$1
  DOCS=$SRC/docs.npy
  case $NAME in
    mda)  # MDA queries are searched against the MS MARCO passages with Snowflake (same index).
      QUERIES=$DER/mda/queries.npy; LOG=$DER/mda/train_queries.npy; GT=$DER/mda/groundtruth.tsv
      DOC=$IDX/snowflake_hnsw_m32_efc500 ;;
    mdav2)
      QUERIES=$DER/mda/queries.npy; LOG=$DER/mda/train_queries.npy; GT=$DER/mda/groundtruth_v2.tsv
      DOC=$IDX/msmarcov2_hnsw_m32_efc500 ;;
    msmarcov2)
      QUERIES=$SRC/queries.npy; LOG=$DATA/snowflake/train_queries.npy; GT=$DER/$NAME/groundtruth.tsv
      DOC=$IDX/msmarcov2_hnsw_m32_efc500 ;;
    *)
      QUERIES=$SRC/queries.npy; LOG=$SRC/train_queries.npy; GT=$DER/$NAME/groundtruth.tsv
      DOC=$IDX/${NAME}_hnsw_m32_efc500 ;;
  esac
  # The MS MARCO v2 index (~320 GB) may not fit in one NUMA node: interleave its memory.
  if [[ -n ${V2_NODES:-} && ( $NAME == msmarcov2 || $NAME == mdav2 ) ]]; then
    NUMA=(numactl --physcpubind="$TIMING_CPU" --interleave="$V2_NODES")
  fi
  PCA_DIM=$((DIM / 4))
  ROUTER=$IDX/${NAME}_router_pca${PCA_DIM}_kep10
}
