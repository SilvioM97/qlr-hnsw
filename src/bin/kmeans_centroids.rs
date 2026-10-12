//! `kmeans_centroids` — k-means centroids of a random sample of a dense dataset (E3).
//!
//! Samples `--sample` rows of `--data-file` uniformly without replacement (seeded), trains
//! vectorium's k-means (spherical by default, since the datasets are searched by dot product)
//! and writes the `k` centroids as an f32 npy file.
//!
//! Assignment uses `train_with_index` with a kANNolo HNSW over the current centroids, rebuilt
//! every iteration: flat assignment of millions of points to tens of thousands of centroids is
//! memory-bound and takes hours.

use std::time::Instant;

use clap::Parser;
use ndarray::Array2;
use ndarray_npy::write_npy;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::index::sample;
use vectorium::distances::SquaredEuclideanDistance;
use vectorium::encoders::dense_scalar::PlainDenseQuantizer;
use vectorium::readers::read_npy_f32;
use vectorium::{Dataset, DenseDataset, KMeansBuilder, PlainDenseDataset};

use kannolo::graph::Graph;
use kannolo::hnsw::{HNSW, HNSWBuildConfiguration, HNSWSearchConfiguration};

#[derive(Parser, Debug)]
struct Args {
    #[clap(long)]
    data_file: String,
    #[clap(long, default_value_t = 2_621_440)]
    sample: usize,
    #[clap(short, long)]
    k: usize,
    #[clap(long, default_value_t = 10)]
    n_iter: usize,
    #[clap(long, default_value_t = 42)]
    seed: u64,
    /// Plain (non-spherical) k-means.
    #[clap(long)]
    euclidean: bool,
    #[clap(short, long)]
    output: String,
}

fn main() {
    let args = Args::parse();
    let t = Instant::now();
    let data = read_npy_f32::<SquaredEuclideanDistance>(&args.data_file).expect("reading data");
    let dim = data.input_dim();
    let n = data.len();
    let values = data.values();
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut rows = sample(&mut rng, n, args.sample.min(n)).into_vec();
    rows.sort_unstable();
    let mut sampled = Vec::with_capacity(rows.len() * dim);
    for &r in &rows {
        sampled.extend_from_slice(&values[r * dim..(r + 1) * dim]);
    }
    drop(data);
    let ds = DenseDataset::from_raw(
        sampled.into_boxed_slice(),
        rows.len(),
        PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(dim),
    );
    println!("[kmeans] sampled {} of {n} rows in {:.1}s", rows.len(), t.elapsed().as_secs_f64());

    let t = Instant::now();
    let km = KMeansBuilder::new()
        .n_iter(args.n_iter)
        .seed(Some(args.seed))
        .spherical(!args.euclidean)
        .verbose(true)
        .build();
    let build = |c: PlainDenseDataset<f32, SquaredEuclideanDistance>| {
        let cfg = HNSWBuildConfiguration::default().with_num_neighbors(32).with_ef_construction(200);
        HNSW::<_, Graph>::build_index(c, &cfg)
    };
    let search = HNSWSearchConfiguration::default().with_ef_search(100);
    let centroids = km.train_with_index(&ds, args.k, None, build, &search);
    println!("[kmeans] {} centroids in {:.1}s", centroids.len(), t.elapsed().as_secs_f64());

    let arr = Array2::from_shape_vec((centroids.len(), dim), centroids.values().to_vec()).unwrap();
    write_npy(&args.output, &arr).expect("writing centroids");
}
