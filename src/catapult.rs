//! CatapultDB-style entry points on HNSW, reimplemented from Abuzakuk et al., "Catapults to the
//! Rescue: Accelerating Vector Search by Exploiting Query Locality" (arXiv:2603.02164,
//! Algorithm 2), whose code is not public. The original runs on DiskANN (Vamana); here the
//! underlying index is the same HNSW used by QLR.
//!
//! - `L` random hyperplanes with standard-normal normals, through the origin, hash the raw
//!   query to one of `2^L` buckets (exact match, no multi-probe).
//! - A bucket holds up to `b` *catapults*: graph nodes that were the best result of earlier
//!   queries with the same hash. The starting points of a search are the bucket contents plus
//!   the index's default entry (DiskANN's medoid; for HNSW, the greedy descent through the upper
//!   levels, [`crate::hnsw::HNSW::ground_entry`]). The beam width is not changed.
//! - After every query, its best result is appended to its bucket; a full bucket evicts its
//!   least-recently-used entry. A destination already in the bucket is moved to the
//!   most-recent position (the paper does not detail duplicates; this keeps LRU semantics and
//!   avoids wasting slots on repeats).

use std::collections::VecDeque;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vectorium::VectorId;

use crate::qlr::dot_f32;

pub struct Catapults {
    dim: usize,
    bits: usize,
    capacity: usize,
    /// `bits` hyperplane normals, row-major (`bits x dim`).
    planes: Vec<f32>,
    buckets: Vec<VecDeque<VectorId>>,
}

impl Catapults {
    /// Empty catapults over `dim`-dimensional queries, `2^bits` buckets of `capacity` entries.
    pub fn new(dim: usize, bits: usize, capacity: usize, seed: u64) -> Self {
        assert!(bits > 0 && bits <= 24, "unsupported number of hyperplanes: {bits}");
        let mut rng = StdRng::seed_from_u64(seed);
        // Standard normal by Box–Muller.
        let planes = (0..bits * dim)
            .map(|_| {
                let u1: f64 = rng.r#gen::<f64>().max(f64::MIN_POSITIVE);
                let u2: f64 = rng.r#gen::<f64>();
                ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
            })
            .collect();
        Self { dim, bits, capacity, planes, buckets: vec![VecDeque::new(); 1 << bits] }
    }

    /// LSH code of `query`: bit `i` is set iff `<query, r_i> >= 0`.
    pub fn hash(&self, query: &[f32]) -> usize {
        debug_assert_eq!(query.len(), self.dim);
        (0..self.bits).fold(0usize, |h, i| {
            let r = &self.planes[i * self.dim..(i + 1) * self.dim];
            h | (((dot_f32(r, query) >= 0.0) as usize) << i)
        })
    }

    /// Catapults of bucket `h`, least recent first.
    pub fn bucket(&self, h: usize) -> &VecDeque<VectorId> {
        &self.buckets[h]
    }

    /// Records `dest` (the best result of a query hashed to `h`) as the most recent catapult.
    pub fn insert(&mut self, h: usize, dest: VectorId) {
        let b = &mut self.buckets[h];
        if let Some(p) = b.iter().position(|&x| x == dest) {
            b.remove(p);
        }
        b.push_back(dest);
        if b.len() > self.capacity {
            b.pop_front();
        }
    }

    /// Number of non-empty buckets.
    pub fn occupied(&self) -> usize {
        self.buckets.iter().filter(|b| !b.is_empty()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_eviction_and_duplicates() {
        let mut c = Catapults::new(4, 2, 3, 7);
        for d in [1, 2, 3, 2, 4] {
            c.insert(0, d);
        }
        // 1 was evicted; 2 moved to the most-recent slot before 4 arrived.
        assert_eq!(c.bucket(0).iter().copied().collect::<Vec<_>>(), vec![3, 2, 4]);
    }

    #[test]
    fn hash_is_sign_pattern() {
        let c = Catapults::new(3, 4, 1, 1);
        let q = [0.3f32, -1.0, 2.0];
        let neg: Vec<f32> = q.iter().map(|x| -x).collect();
        // Opposite vectors have complementary codes (no component is exactly zero here).
        assert_eq!(c.hash(&q) ^ c.hash(&neg), (1 << 4) - 1);
    }
}
