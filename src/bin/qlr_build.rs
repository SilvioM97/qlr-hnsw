//! `qlr_build` — build a QLR router (`I_Q` + `EP` + projection) over an existing document index.
//!
//! The document index is the one written by `hnsw_build`. The router is
//! written to its own file; the document index is never rewritten.
//!
//! The PCA projection is fitted outside (scripts/fit_pca.py) and passed as two npy files,
//! `{prefix}_W.npy` (`out_dim x in_dim`, f32) and `{prefix}_b.npy` (`out_dim`, f32).
//! Build timings are printed and, with `--times-json`, written as JSON.

use std::io::Write;
use std::time::Instant;

use clap::Parser;
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use vectorium::IndexSerializer;
use vectorium::distances::DotProduct;
use vectorium::readers::read_npy_f32;

use kannolo::qlr::{Projection, QlrRouter, doc_space_usage_bytes, load_doc_index};

#[derive(Parser, Debug)]
struct Args {
    /// Document HNSW (dot product, f16, fixed-degree).
    #[clap(long)]
    doc_index: String,
    /// Log vectors (npy f32, n x d): historical queries, or centroids / documents for E3.
    #[clap(long, default_value = "")]
    log_file: String,
    /// Optional npy (int64) of row indices of `log_file` to use (subsampling, E3/E6).
    #[clap(long)]
    log_rows: Option<String>,
    /// Prefix of the projection files `{prefix}_W.npy`, `{prefix}_b.npy`. No PCA when absent.
    #[clap(long)]
    projection_prefix: Option<String>,
    #[clap(long, default_value_t = 10)]
    k_ep: usize,
    #[clap(long, default_value_t = 32)]
    query_index_m: usize,
    #[clap(long, default_value_t = 1000)]
    query_index_ef_construction: usize,
    #[clap(long, default_value_t = 200)]
    training_search_ef: usize,
    #[clap(short, long)]
    output: String,
    /// Write build timings and sizes here (JSON).
    #[clap(long)]
    times_json: Option<String>,
}

fn main() {
    let args = Args::parse();

    let t = Instant::now();
    let doc = load_doc_index(&args.doc_index);
    println!("[qlr_build] doc index loaded in {:.1}s", t.elapsed().as_secs_f64());

    let log = read_npy_f32::<DotProduct>(&args.log_file).expect("reading log file");
    let dim = vectorium::Dataset::input_dim(&log);
    let mut values: Vec<f32> = vectorium::DenseDataset::values(&log).to_vec();
    drop(log);
    if let Some(rows_path) = &args.log_rows {
        let rows: Array1<i64> = read_npy(rows_path).expect("reading log rows");
        let mut sel = Vec::with_capacity(rows.len() * dim);
        for &r in rows.iter() {
            let r = r as usize;
            sel.extend_from_slice(&values[r * dim..(r + 1) * dim]);
        }
        values = sel;
    }
    let n = values.len() / dim;
    println!("[qlr_build] {n} log vectors x {dim} dim");

    let projection = args.projection_prefix.as_ref().map(|p| {
        let w: Array2<f32> = read_npy(format!("{p}_W.npy")).expect("reading W");
        let b: Array1<f32> = read_npy(format!("{p}_b.npy")).expect("reading b");
        let (out_dim, in_dim) = w.dim();
        assert_eq!(in_dim, dim, "projection input dim must match log dim");
        Projection::new(in_dim, out_dim, w.iter().copied().collect(), b.to_vec())
    });

    let (router, times) = QlrRouter::build(
        &doc,
        &values,
        dim,
        projection,
        args.k_ep,
        args.query_index_m,
        args.query_index_ef_construction,
        args.training_search_ef,
    );
    let (iq_b, ep_b, proj_b) = router.space_usage_parts();
    let doc_b = doc_space_usage_bytes(&doc);
    println!(
        "[qlr_build] project {:.1}s, I_Q {:.1}s, EP {:.1}s; Q3 = {}",
        times.project_s, times.query_index_s, times.ep_s, router.q3_score
    );
    println!("[qlr_build] bytes: I_Q {iq_b}, EP {ep_b}, PCA {proj_b}, I_D {doc_b}");

    router.save_index(&args.output).expect("saving router");

    if let Some(path) = &args.times_json {
        let mut f = std::fs::File::create(path).expect("creating times json");
        writeln!(
            f,
            "{{\"n_log\": {n}, \"dim\": {dim}, \"k_ep\": {}, \"project_s\": {}, \"query_index_s\": {}, \"ep_s\": {}, \"q3\": {}, \"iq_bytes\": {iq_b}, \"ep_bytes\": {ep_b}, \"pca_bytes\": {proj_b}, \"doc_bytes\": {doc_b}, \"threads\": {}}}",
            args.k_ep,
            times.project_s,
            times.query_index_s,
            times.ep_s,
            router.q3_score,
            rayon::current_num_threads()
        )
        .unwrap();
    }
}
