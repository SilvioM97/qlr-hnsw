//! Query-Log Routing (QLR) for HNSW.
//!
//! A small auxiliary index `I_Q` over (optionally PCA-projected) historical query vectors, plus a
//! lookup table `EP` mapping each historical query to the top-`k_ep` documents the document index
//! returned for it. At query time the nearest historical queries are found in `I_Q`, their `EP`
//! entries are scored exactly against the query, and if the best normalized score clears a
//! threshold the ground level of the document HNSW is searched from those entries, with a beam
//! width that shrinks with the routing confidence. Otherwise a plain HNSW search is run.
//!
//! Port of the `qlr-hnsw` crate (built on kannolo_private `438fde7`) with the same search
//! semantics. Differences: the router is stored apart from the document index, `EP` keeps only
//! `u32` ids, the PCA projection is a single folded `f32` matrix applied per query, and every
//! query records timing and routing statistics.
//!
//! Specialized to the configuration used in the paper: dot-product, `f16` storage, `f32`
//! queries, fixed-degree graphs.

use std::time::Instant;

use half::f16;
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use vectorium::IndexSerializer;
use vectorium::core::index::Index;
use vectorium::distances::{Distance, DotProduct, dot_product_dense};
use vectorium::encoders::dense_scalar::PlainDenseQuantizer;
use vectorium::vector::DenseVectorView;
use vectorium::{DenseDataset, ScoredVector, VectorId};

use crate::graph::GraphFixedDegree;
use crate::hnsw::{HNSW, HNSWBuildConfiguration, HNSWSearchConfiguration};

/// Dataset type of both indexes: `f16` storage, `f32` queries, dot product.
pub type QlrDataset = DenseDataset<PlainDenseQuantizer<f16, DotProduct>>;
/// Index type of both `I_D` and `I_Q`.
pub type QlrHnsw = HNSW<QlrDataset, GraphFixedDegree>;

/// Default minimum beam width of the adaptive scheme (`ef_min` in the paper); see
/// [`QlrParams::ef_min`].
pub const EF_MIN: usize = 10;

/// Loads a document index written by `hnsw_build`.
pub fn load_doc_index(path: &str) -> QlrHnsw {
    <QlrHnsw as IndexSerializer>::load_index(path)
        .unwrap_or_else(|e| panic!("cannot load document index {path}: {e:?}"))
}

/// Converts a row-major `f32` matrix into a [`QlrDataset`].
pub fn dataset_from_f32(values: &[f32], dim: usize) -> QlrDataset {
    assert!(dim > 0 && values.len() % dim == 0);
    let data: Box<[f16]> = values.iter().map(|&x| f16::from_f32(x)).collect();
    DenseDataset::from_raw(data, values.len() / dim, PlainDenseQuantizer::new(dim))
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    // vectorium's SIMD dot product (the f32 path of the HNSW search itself).
    dot_product_dense(DenseVectorView::new(a), DenseVectorView::new(b)).distance()
}

#[inline]
fn l2_norm(x: &[f32]) -> f32 {
    dot(x, x).sqrt()
}

/// Affine projection `y = W x + b` (`W` is `out_dim x in_dim`, row-major).
///
/// A PCA fitted on standardized features is folded into this form: with per-feature mean `m`,
/// scale `s` and orthonormal components `R` (`in_dim x out_dim`), `W = R^T diag(1/s)` and
/// `b = -W m`. This matches the transform of the `efficient_pca` crate used by the old code.
#[derive(Serialize, Deserialize, Clone)]
pub struct Projection {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

impl Projection {
    pub fn new(in_dim: usize, out_dim: usize, weights: Vec<f32>, bias: Vec<f32>) -> Self {
        assert_eq!(weights.len(), in_dim * out_dim);
        assert_eq!(bias.len(), out_dim);
        Self {
            in_dim,
            out_dim,
            weights,
            bias,
        }
    }

    #[inline]
    pub fn project_into(&self, x: &[f32], out: &mut Vec<f32>) {
        out.clear();
        out.extend(
            self.weights
                .chunks_exact(self.in_dim)
                .zip(&self.bias)
                .map(|(row, b)| dot(row, x) + b),
        );
    }

    pub fn space_usage_bytes(&self) -> usize {
        (self.weights.len() + self.bias.len()) * std::mem::size_of::<f32>()
    }
}

/// `EP` lookup table: `k_ep` document ids per historical query, best first, `u32::MAX`-padded.
#[derive(Serialize, Deserialize, Clone)]
pub struct EpTable {
    pub k_ep: usize,
    pub ids: Box<[u32]>,
    /// Normalized top-1 score `<q_l, d_1> / ||q_l||` of each historical query.
    pub best_scores: Box<[f32]>,
}

impl EpTable {
    /// Searches `doc` for every row of `log` (full-dimensional, row-major) in parallel.
    pub fn build(doc: &QlrHnsw, log: &[f32], dim: usize, k_ep: usize, ef: usize) -> Self {
        let n = log.len() / dim;
        let cfg = HNSWSearchConfiguration::default().with_ef_search(ef);
        let rows: Vec<(Vec<u32>, f32)> = (0..n)
            .into_par_iter()
            .progress_count(n as u64)
            .map(|i| {
                let q = &log[i * dim..(i + 1) * dim];
                let res = doc.search(DenseVectorView::new(q), k_ep, &cfg);
                let norm = l2_norm(q);
                let best = res
                    .first()
                    .map(|sv| {
                        let s = sv.distance.distance();
                        if norm > 0.0 { s / norm } else { s }
                    })
                    .unwrap_or(f32::NEG_INFINITY);
                let mut ids: Vec<u32> = res.iter().map(|sv| sv.vector as u32).collect();
                ids.resize(k_ep, u32::MAX);
                (ids, best)
            })
            .collect();
        let mut ids = Vec::with_capacity(n * k_ep);
        let mut best_scores = Vec::with_capacity(n);
        for (r, b) in rows {
            ids.extend_from_slice(&r);
            best_scores.push(b);
        }
        Self {
            k_ep,
            ids: ids.into_boxed_slice(),
            best_scores: best_scores.into_boxed_slice(),
        }
    }

    #[inline]
    pub fn entries(&self, log_id: usize, take: usize) -> impl Iterator<Item = u32> + '_ {
        let row = &self.ids[log_id * self.k_ep..(log_id + 1) * self.k_ep];
        row.iter().take(take).copied().filter(|&d| d != u32::MAX)
    }

    pub fn len(&self) -> usize {
        self.best_scores.len()
    }

    pub fn is_empty(&self) -> bool {
        self.best_scores.is_empty()
    }

    /// 75th percentile (top-25% boundary) of the best scores: `s_max` in the paper.
    pub fn q3(&self) -> f32 {
        let mut s: Vec<f32> = self
            .best_scores
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        if s.is_empty() {
            return 1.0;
        }
        s.sort_unstable_by(|a, b| b.total_cmp(a));
        s[((s.len() as f32 * 0.25) as usize).min(s.len() - 1)]
    }

    pub fn space_usage_bytes(&self) -> usize {
        self.ids.len() * std::mem::size_of::<u32>()
    }
}

/// The QLR router: `I_Q`, `EP`, the projection and `s_max`. Serialized apart from `I_D`.
#[derive(Serialize, Deserialize)]
pub struct QlrRouter {
    pub query_index: QlrHnsw,
    pub ep: EpTable,
    pub projection: Option<Projection>,
    pub q3_score: f32,
}

/// Wall-clock breakdown of a router build, in seconds.
#[derive(Debug, Default, Clone, Copy)]
pub struct BuildTimes {
    pub project_s: f64,
    pub query_index_s: f64,
    pub ep_s: f64,
}

impl IndexSerializer for QlrRouter {}

impl QlrRouter {
    /// Builds `I_Q` over the (projected) log vectors and populates `EP` by searching `doc`.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        doc: &QlrHnsw,
        log: &[f32],
        dim: usize,
        projection: Option<Projection>,
        k_ep: usize,
        query_index_m: usize,
        query_index_ef_construction: usize,
        training_search_ef: usize,
    ) -> (Self, BuildTimes) {
        let mut times = BuildTimes::default();
        let n = log.len() / dim;

        let t = Instant::now();
        let (reduced, rdim) = match &projection {
            Some(p) => {
                assert_eq!(p.in_dim, dim);
                let out: Vec<f32> = (0..n)
                    .into_par_iter()
                    .flat_map_iter(|i| {
                        let mut y = Vec::with_capacity(p.out_dim);
                        p.project_into(&log[i * dim..(i + 1) * dim], &mut y);
                        y
                    })
                    .collect();
                (out, p.out_dim)
            }
            None => (log.to_vec(), dim),
        };
        times.project_s = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let cfg = HNSWBuildConfiguration::default()
            .with_num_neighbors(query_index_m)
            .with_ef_construction(query_index_ef_construction);
        let query_index = HNSW::build_index(dataset_from_f32(&reduced, rdim), &cfg);
        times.query_index_s = t.elapsed().as_secs_f64();
        drop(reduced);

        let t = Instant::now();
        let ep = EpTable::build(doc, log, dim, k_ep, training_search_ef);
        times.ep_s = t.elapsed().as_secs_f64();

        let q3_score = ep.q3();
        (
            Self {
                query_index,
                ep,
                projection,
                q3_score,
            },
            times,
        )
    }

    /// Bytes of `I_Q` (graph + vectors), `EP` and the projection matrix.
    pub fn space_usage_parts(&self) -> (usize, usize, usize) {
        (
            self.query_index.space_usage_bytes(),
            self.ep.space_usage_bytes(),
            self.projection.as_ref().map_or(0, Projection::space_usage_bytes),
        )
    }
}

/// What a routed query does once it clears the threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutedAction {
    /// Seeded ground-level beam search (QLR).
    Search,
    /// Return `EP(q_1)` directly, no search (approximate-cache baseline, E8).
    ReturnCached,
}

/// Search-time knobs.
#[derive(Debug, Clone, Copy)]
pub struct QlrParams {
    /// Number of historical queries retrieved from `I_Q` (`k'` in the paper; it is also the
    /// `ef_search` of `I_Q`, as in the old code).
    pub k_prime: usize,
    /// `EP` entries taken per retrieved historical query (`<= k_ep`).
    pub ep_per_query: usize,
    /// Default beam width `ef` of the document index.
    pub ef: usize,
    /// Routing threshold `th` on the normalized score.
    pub th: f32,
    /// Shrink `ef` with routing confidence; `false` = seeding only (E2b).
    pub adaptive: bool,
    /// Beam width for routed queries with `s >= s_max`. A floor of 10 caps the recall of the
    /// most confident queries, which binds only at the highest accuracy targets.
    pub ef_min: usize,
    /// Reuse the routing distances as seed distances instead of recomputing them. The submitted
    /// code recomputed them; results are identical, only latency changes.
    pub reuse_seed_scores: bool,
    pub action: RoutedAction,
}

/// Per-query instrumentation. Times are in nanoseconds.
#[derive(Debug, Default, Clone, Copy)]
pub struct QueryStats {
    pub routed: bool,
    /// Normalized routing score `s` (NaN if no candidates).
    pub score: f32,
    pub ef_used: usize,
    pub n_seeds: usize,
    pub proj_ns: u64,
    /// `I_Q` search (or window scan).
    pub iq_ns: u64,
    /// Gathering + exact scoring of the `EP` candidates.
    pub score_ns: u64,
    /// Beam search on `I_D` (seeded or fallback).
    pub beam_ns: u64,
}

impl QueryStats {
    pub fn router_ns(&self) -> u64 {
        self.proj_ns + self.iq_ns
    }
}

/// Linear interpolation of the beam width between `ef_min` (score at or above `q3`) and `ef`
/// (score at `th`), as in `qlr-hnsw::adaptive` (dot-product direction).
#[must_use]
pub fn adaptive_ef(score: f32, ef: usize, q3: f32, th: f32, ef_min: usize) -> usize {
    let ef_max = ef.max(ef_min);
    // Same branch order as the old `Ord`-based clamp: at least as good as q3 -> q3 (even when
    // th > q3), else worse than th -> th, else the score itself.
    let used = if score >= q3 {
        q3
    } else if score < th {
        th
    } else {
        score
    };
    let denom = th - q3;
    if denom == 0.0 {
        return ef_min;
    }
    let t = (used - q3) / denom;
    (ef_min as f32 + (ef_max - ef_min) as f32 * t) as usize
}

/// Second half of routing, shared by the global router and the window variant: score the `EP`
/// entries of `log_hits`, gate on `th`, and run the seeded or the fallback search.
#[allow(clippy::too_many_arguments)]
pub fn route_from_hits(
    doc: &QlrHnsw,
    ep: &EpTable,
    q3: f32,
    query: &[f32],
    log_hits: &[usize],
    k: usize,
    p: &QlrParams,
    stats: &mut QueryStats,
) -> Vec<ScoredVector<DotProduct>> {
    let qv = DenseVectorView::new(query);
    let doc_cfg = HNSWSearchConfiguration::default().with_ef_search(p.ef);

    let t = Instant::now();
    let candidates: Vec<VectorId> = log_hits
        .iter()
        .flat_map(|&l| ep.entries(l, p.ep_per_query))
        .map(|d| d as VectorId)
        .collect();
    if candidates.is_empty() {
        stats.score_ns = t.elapsed().as_nanos() as u64;
        stats.score = f32::NAN;
        stats.ef_used = p.ef;
        let t = Instant::now();
        let res = doc.search(qv, k, &doc_cfg);
        stats.beam_ns = t.elapsed().as_nanos() as u64;
        return res;
    }
    let scored = doc.score_ids(qv, &candidates, &());
    let norm = l2_norm(query);
    let raw = scored[0].distance.distance();
    let s = if norm > 0.0 { raw / norm } else { raw };
    stats.score_ns = t.elapsed().as_nanos() as u64;
    stats.score = s;
    stats.n_seeds = scored.len();

    let t = Instant::now();
    let res = if s >= p.th {
        stats.routed = true;
        match p.action {
            RoutedAction::ReturnCached => {
                stats.ef_used = 0;
                ep.entries(log_hits[0], k)
                    .map(|d| ScoredVector {
                        distance: DotProduct(0.0),
                        vector: d as VectorId,
                    })
                    .collect()
            }
            RoutedAction::Search => {
                let ef = if p.adaptive {
                    adaptive_ef(s, p.ef, q3, p.th, p.ef_min)
                } else {
                    p.ef
                };
                stats.ef_used = ef;
                let cfg = HNSWSearchConfiguration::default().with_ef_search(ef);
                if p.reuse_seed_scores {
                    doc.search_from_scored_seeds(qv, &scored, k, &cfg)
                } else {
                    let ids: Vec<VectorId> = scored.iter().map(|sv| sv.vector).collect();
                    doc.search_from_entry_points(qv, &ids, k, &cfg)
                }
            }
        }
    } else {
        stats.ef_used = p.ef;
        doc.search(qv, k, &doc_cfg)
    };
    stats.beam_ns = t.elapsed().as_nanos() as u64;
    res
}

impl QlrRouter {
    /// Routed search. `scratch` holds the projected query between calls.
    pub fn search(
        &self,
        doc: &QlrHnsw,
        query: &[f32],
        k: usize,
        p: &QlrParams,
        scratch: &mut Vec<f32>,
        stats: &mut QueryStats,
    ) -> Vec<ScoredVector<DotProduct>> {
        *stats = QueryStats::default();

        let t = Instant::now();
        let short: &[f32] = match &self.projection {
            Some(proj) => {
                proj.project_into(query, scratch);
                scratch
            }
            None => query,
        };
        stats.proj_ns = t.elapsed().as_nanos() as u64;

        let t = Instant::now();
        let q_cfg = HNSWSearchConfiguration::default().with_ef_search(p.k_prime);
        let hits = self
            .query_index
            .search(DenseVectorView::new(short), p.k_prime, &q_cfg);
        let log_hits: Vec<usize> = hits
            .iter()
            .take(p.k_prime)
            .map(|sv| sv.vector as usize)
            .collect();
        stats.iq_ns = t.elapsed().as_nanos() as u64;

        route_from_hits(doc, &self.ep, self.q3_score, query, &log_hits, k, p, stats)
    }
}

/// Exact top-`k'` over a window of projected stream vectors (rows `lo..hi` of `vectors`).
/// Used by the session-window router (E4). Returns stream row ids, best first.
pub fn window_top(vectors: &[f32], dim: usize, lo: usize, hi: usize, q: &[f32], kp: usize) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = (lo..hi)
        .map(|r| (dot(&vectors[r * dim..(r + 1) * dim], q), r))
        .collect();
    let kp = kp.min(scored.len());
    if kp == 0 {
        return Vec::new();
    }
    scored.select_nth_unstable_by(kp - 1, |a, b| b.0.total_cmp(&a.0));
    scored.truncate(kp);
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    scored.into_iter().map(|(_, r)| r).collect()
}

/// Public wrapper of the internal dot product, for binaries.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b)
}

/// Bytes of a document index (vectors + graph).
pub fn doc_space_usage_bytes(doc: &QlrHnsw) -> usize {
    doc.space_usage_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_endpoints() {
        assert_eq!(adaptive_ef(0.85, 200, 0.85, 0.6, EF_MIN), EF_MIN);
        assert_eq!(adaptive_ef(0.60, 200, 0.85, 0.6, EF_MIN), 200);
        assert_eq!(adaptive_ef(0.99, 200, 0.85, 0.6, EF_MIN), EF_MIN);
        assert_eq!(adaptive_ef(0.10, 200, 0.85, 0.6, EF_MIN), 200);
        assert_eq!(adaptive_ef(0.7, 200, 0.7, 0.7, EF_MIN), EF_MIN);
        // th above q3: any routed query is already "better than q3".
        assert_eq!(adaptive_ef(0.75, 200, 0.6, 0.7, EF_MIN), EF_MIN);
    }

    #[test]
    fn dot_matches_naive() {
        let a: Vec<f32> = (0..37).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..37).map(|i| 1.0 - i as f32 * 0.25).collect();
        let naive: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - naive).abs() < 1e-3);
    }

    #[test]
    fn window_top_orders_best_first() {
        let v = vec![1.0, 0.0, 0.0, 1.0, 0.7, 0.7];
        assert_eq!(window_top(&v, 2, 0, 3, &[1.0, 0.1], 2), vec![0, 2]);
        assert!(window_top(&v, 2, 1, 1, &[1.0, 0.1], 2).is_empty());
    }
}
