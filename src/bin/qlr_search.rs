//! `qlr_search` — single-threaded latency/accuracy sweeps for HNSW, QLR and the window router.
//!
//! The document index is loaded once and every configuration of the grid is run in the same
//! process. Each configuration appends one CSV row to `--out-csv`; with `--per-query-dir`, a
//! per-query file (routing decision, score, ef used, time breakdown, recall) is written too.
//!
//! Latency is wall-clock around the whole query: projection, `I_Q` search, scoring, beam search.
//! Accuracy@k is |result ∩ GT top-k| / k, averaged over queries.
//!
//! Modes:
//! - `hnsw`:   plain HNSW over `--ef-list`.
//! - `qlr`:    routed search over `--kprime-list x --th-list x --ef-list x adaptive`, or over the
//!   configurations listed in `--configs-file`.
//! - `window`: E4. The router is the last `w` stream queries before each test query
//!   (`--stream`, `--test-stream-pos`), with EP built on the fly by searching `I_D`. The
//!   projection and `s_max` come from `--router` (the global one).
//! - `catapult`: CatapultDB-style baseline ([`kannolo::catapult`]) over
//!   `--lsh-bits-list x --bucket-size-list x --catapult-warmup-list x --ef-list`. Catapults are
//!   updated online, after every query, in `--query-order`. With warm-up `log`, the query log
//!   (`--log-file`, rows aligned with the EP table of `--router`) is first streamed through the
//!   same update rule, each log query's best result being its first `EP` entry.
//!   `--catapult-entry-list`: `descent` seeds the ground level with the bucket plus HNSW's
//!   default entry (the upper-level descent, paid by every query); `fallback` seeds it with the
//!   bucket only, and descends the hierarchy only when the bucket is empty.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use ndarray::Array1;
use ndarray_npy::read_npy;
use vectorium::IndexSerializer;
use vectorium::core::index::Index;
use vectorium::distances::DotProduct;
use vectorium::readers::read_npy_f32;
use vectorium::vector::DenseVectorView;

use kannolo::catapult::Catapults;
use kannolo::hnsw::HNSWSearchConfiguration;
use kannolo::qlr::{
    EpTable, QlrHnsw, QlrParams, QlrRouter, QueryStats, RoutedAction, load_doc_index,
    route_from_hits, window_top,
};

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum Mode {
    Hnsw,
    Qlr,
    Window,
    Catapult,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum Adaptive {
    True,
    False,
    Both,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum Action {
    Search,
    Cache,
}

#[derive(Parser, Debug)]
struct Args {
    #[clap(long, value_enum)]
    mode: Mode,
    #[clap(long)]
    doc_index: String,
    #[clap(long)]
    queries: String,
    /// Ground truth TSV: query_idx, doc_id, rank, score.
    #[clap(long)]
    gt: String,
    #[clap(long)]
    router: Option<String>,
    /// Use this s_max (Q3) instead of the router's own. E3 keeps the mechanism fixed across
    /// entry-point sources: a router over documents has s_max = 1 (each document is its own top-1).
    #[clap(long)]
    q3_override: Option<f32>,
    #[clap(short, long, default_value_t = 10)]
    k: usize,
    #[clap(long, value_delimiter = ',', default_value = "10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200")]
    ef_list: Vec<usize>,
    #[clap(long, value_delimiter = ',', default_value = "10,20")]
    kprime_list: Vec<usize>,
    #[clap(long, value_delimiter = ',', default_value = "0.3,0.4,0.5,0.6,0.7")]
    th_list: Vec<f32>,
    /// EP entries per retrieved log query (default: k_ep of the router).
    #[clap(long)]
    ep_per_query: Option<usize>,
    #[clap(long, value_enum, default_value_t = Adaptive::True)]
    adaptive: Adaptive,
    /// Minimum beam width(s) of the adaptive scheme.
    #[clap(long, value_delimiter = ',', default_value = "10")]
    ef_min_list: Vec<usize>,
    /// Run exactly the configurations listed in this CSV (header included) instead of the cross
    /// product of the lists. `qlr`: columns `k_prime,th,ef,ef_min,adaptive` (empty `ef_min` =
    /// default, empty `adaptive` = true); `catapult`: `lsh_bits,bucket_size,warmup,entry,ef`.
    #[clap(long)]
    configs_file: Option<String>,
    #[clap(long, value_enum, default_value_t = Action::Search)]
    action: Action,
    /// Reuse routing distances as seed distances (identical results, faster).
    #[clap(long)]
    reuse_seed_scores: bool,
    /// Window sizes for `--mode window`.
    #[clap(long, value_delimiter = ',', default_value = "1,10,100,1000")]
    w_list: Vec<usize>,
    /// Stream vectors (npy f32) for `--mode window`, chronological.
    #[clap(long)]
    stream: Option<String>,
    /// npy int64: stream row of each test query (row i of `--queries`).
    #[clap(long)]
    test_stream_pos: Option<String>,
    /// k_ep for the window EP table.
    #[clap(long, default_value_t = 10)]
    window_k_ep: usize,
    /// Catapult mode: numbers of LSH hyperplanes (2^L buckets).
    #[clap(long, value_delimiter = ',', default_value = "8")]
    lsh_bits_list: Vec<usize>,
    /// Catapult mode: bucket capacities b.
    #[clap(long, value_delimiter = ',', default_value = "40")]
    bucket_size_list: Vec<usize>,
    /// Catapult mode: `none` (empty buckets, as in the paper) and/or `log` (warmed up with the
    /// query log).
    #[clap(long, value_delimiter = ',', default_value = "none,log")]
    catapult_warmup_list: Vec<String>,
    /// Catapult mode: `descent` (bucket + default entry) and/or `fallback` (bucket only; the
    /// default entry only for empty buckets).
    #[clap(long, value_delimiter = ',', default_value = "descent")]
    catapult_entry_list: Vec<String>,
    /// Catapult mode: query-log vectors (npy f32) for the `log` warm-up.
    #[clap(long)]
    log_file: Option<String>,
    #[clap(long, default_value_t = 42)]
    lsh_seed: u64,
    /// Process queries in this order (npy int64 of query rows); default natural order.
    #[clap(long)]
    query_order: Option<String>,
    /// Use only the first n queries (after ordering).
    #[clap(long)]
    limit: Option<usize>,
    /// Untimed warm-up queries before the first configuration.
    #[clap(long, default_value_t = 1000)]
    warmup: usize,
    #[clap(long)]
    out_csv: String,
    #[clap(long)]
    per_query_dir: Option<String>,
    /// Free-form labels written in every row.
    #[clap(long, default_value = "")]
    experiment: String,
    #[clap(long, default_value = "")]
    dataset: String,
    #[clap(long, default_value = "")]
    method: String,
}

const HEADER: &str = "experiment,dataset,method,config,k_prime,ep_per_query,th,ef,adaptive,action,w,accuracy,mean_latency_us,hit_rate,mean_router_us,mean_proj_us,mean_iq_us,mean_score_us,mean_beam_us,mean_ef_used,mean_seeds,n_queries,ef_min";

/// Rows of a CSV file with a header, as column name -> value.
fn read_config_rows(path: &str) -> Vec<std::collections::HashMap<String, String>> {
    let f = BufReader::new(File::open(path).unwrap_or_else(|e| panic!("opening {path}: {e}")));
    let mut lines = f.lines().map(|l| l.expect("reading configs file")).filter(|l| !l.trim().is_empty());
    let header: Vec<String> = lines.next().expect("empty configs file").split(',').map(|c| c.trim().to_string()).collect();
    lines
        .map(|l| header.iter().cloned().zip(l.split(',').map(|v| v.trim().to_string())).collect())
        .collect()
}

fn read_gt(path: &str, k: usize) -> Vec<Vec<u64>> {
    let f = BufReader::new(File::open(path).expect("opening gt"));
    let mut gt: Vec<Vec<(usize, u64)>> = Vec::new();
    for line in f.lines() {
        let line = line.unwrap();
        let mut it = line.split('\t');
        let q: usize = it.next().unwrap().trim().parse().unwrap();
        let d: u64 = it.next().unwrap().trim().parse().unwrap();
        let r: usize = it.next().unwrap().trim().parse().unwrap();
        if gt.len() <= q {
            gt.resize(q + 1, Vec::new());
        }
        gt[q].push((r, d));
    }
    gt.into_iter()
        .map(|mut v| {
            v.sort_unstable();
            v.into_iter().take(k).map(|(_, d)| d).collect()
        })
        .collect()
}

fn read_i64(path: &str) -> Vec<usize> {
    let a: Array1<i64> = read_npy(path).expect("reading int64 npy");
    a.iter().map(|&x| x as usize).collect()
}

struct Row {
    config: String,
    k_prime: String,
    ep_per_query: String,
    th: String,
    ef: usize,
    adaptive: String,
    action: String,
    w: String,
    ef_min: usize,
}

struct Runner<'a> {
    args: &'a Args,
    gt: Vec<Vec<u64>>,
    order: Vec<usize>,
    csv: BufWriter<File>,
}

impl Runner<'_> {
    /// Runs `f` for every query in order, then appends the aggregate row.
    fn run<F>(&mut self, row: Row, mut f: F)
    where
        F: FnMut(usize, &mut QueryStats) -> Vec<u64>,
    {
        let k = self.args.k;
        let n = self.order.len();
        let mut per_query: Vec<(usize, QueryStats, u64, f32)> = Vec::with_capacity(n);
        for &qi in &self.order {
            let mut st = QueryStats::default();
            let t = Instant::now();
            let ids = f(qi, &mut st);
            let total = t.elapsed().as_nanos() as u64;
            let gt: HashSet<u64> = self.gt[qi].iter().copied().collect();
            let hits = ids.iter().take(k).filter(|d| gt.contains(d)).count();
            per_query.push((qi, st, total, hits as f32 / k as f32));
        }
        let nf = n as f64;
        let mean = |g: &dyn Fn(&(usize, QueryStats, u64, f32)) -> f64| -> f64 {
            per_query.iter().map(g).sum::<f64>() / nf
        };
        let acc = mean(&|r| r.3 as f64) * 100.0;
        let lat = mean(&|r| r.2 as f64) / 1000.0;
        let hit = mean(&|r| r.1.routed as u8 as f64);
        let router = mean(&|r| r.1.router_ns() as f64) / 1000.0;
        let proj = mean(&|r| r.1.proj_ns as f64) / 1000.0;
        let iq = mean(&|r| r.1.iq_ns as f64) / 1000.0;
        let score = mean(&|r| r.1.score_ns as f64) / 1000.0;
        let beam = mean(&|r| r.1.beam_ns as f64) / 1000.0;
        let efu = mean(&|r| r.1.ef_used as f64);
        let seeds = mean(&|r| r.1.n_seeds as f64);
        let a = self.args;
        writeln!(
            self.csv,
            "{},{},{},{},{},{},{},{},{},{},{},{:.3},{:.1},{:.4},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{},{}",
            a.experiment, a.dataset, a.method, row.config, row.k_prime, row.ep_per_query, row.th,
            row.ef, row.adaptive, row.action, row.w, acc, lat, hit, router, proj, iq, score, beam,
            efu, seeds, n, row.ef_min
        )
        .unwrap();
        self.csv.flush().unwrap();
        println!("[qlr_search] {} acc {acc:.3} lat {lat:.1}us hit {hit:.3}", row.config);

        if let Some(dir) = &a.per_query_dir {
            std::fs::create_dir_all(dir).unwrap();
            let name = format!(
                "{}/{}_{}_{}.csv",
                dir,
                a.experiment,
                a.dataset,
                row.config.replace(['/', ' '], "_")
            );
            let mut w = BufWriter::new(File::create(name).unwrap());
            writeln!(w, "query,routed,score,ef_used,n_seeds,proj_ns,iq_ns,score_ns,beam_ns,total_ns,recall").unwrap();
            for (qi, st, total, rec) in &per_query {
                writeln!(
                    w,
                    "{qi},{},{},{},{},{},{},{},{},{total},{rec}",
                    st.routed as u8, st.score, st.ef_used, st.n_seeds, st.proj_ns, st.iq_ns,
                    st.score_ns, st.beam_ns
                )
                .unwrap();
            }
        }
    }
}

fn main() {
    let args = Args::parse();

    let t = Instant::now();
    let doc: QlrHnsw = load_doc_index(&args.doc_index);
    println!("[qlr_search] doc index loaded in {:.1}s", t.elapsed().as_secs_f64());

    let queries_ds = read_npy_f32::<DotProduct>(&args.queries).expect("reading queries");
    let dim = vectorium::Dataset::input_dim(&queries_ds);
    let queries: Vec<f32> = vectorium::DenseDataset::values(&queries_ds).to_vec();
    drop(queries_ds);
    let nq = queries.len() / dim;
    let q = |i: usize| &queries[i * dim..(i + 1) * dim];

    let gt = read_gt(&args.gt, args.k);
    let mut order: Vec<usize> = match &args.query_order {
        Some(p) => read_i64(p),
        None => (0..nq).collect(),
    };
    if let Some(l) = args.limit {
        order.truncate(l);
    }
    assert!(gt.len() >= nq, "ground truth covers {} queries, need {nq}", gt.len());

    let router: Option<QlrRouter> = args.router.as_ref().map(|p| {
        let t = Instant::now();
        let mut r = <QlrRouter as IndexSerializer>::load_index(p).expect("loading router");
        if let Some(q3) = args.q3_override {
            println!("[qlr_search] s_max overridden: {} -> {q3}", r.q3_score);
            r.q3_score = q3;
        }
        println!("[qlr_search] router loaded in {:.1}s (Q3 = {})", t.elapsed().as_secs_f64(), r.q3_score);
        r
    });

    let new_file = !Path::new(&args.out_csv).exists();
    let mut csv = BufWriter::new(
        OpenOptions::new().create(true).append(true).open(&args.out_csv).unwrap(),
    );
    if new_file {
        writeln!(csv, "{HEADER}").unwrap();
    }

    // Warm-up: plain HNSW searches, untimed.
    let warm_cfg = HNSWSearchConfiguration::default().with_ef_search(100);
    for &qi in order.iter().take(args.warmup) {
        let _ = doc.search(DenseVectorView::new(q(qi)), args.k, &warm_cfg);
    }

    let mut runner = Runner {
        args: &args,
        gt,
        order: order.clone(),
        csv,
    };

    let adaptive_list: Vec<bool> = match args.adaptive {
        Adaptive::True => vec![true],
        Adaptive::False => vec![false],
        Adaptive::Both => vec![true, false],
    };
    let action = match args.action {
        Action::Search => RoutedAction::Search,
        Action::Cache => RoutedAction::ReturnCached,
    };
    let action_s = format!("{:?}", args.action).to_lowercase();

    match args.mode {
        Mode::Hnsw => {
            for &ef in &args.ef_list {
                let cfg = HNSWSearchConfiguration::default().with_ef_search(ef);
                runner.run(
                    Row {
                        config: format!("hnsw_ef{ef}"),
                        k_prime: String::new(),
                        ep_per_query: String::new(),
                        th: String::new(),
                        ef,
                        adaptive: String::new(),
                        action: String::new(),
                        w: String::new(),
                        ef_min: 0,
                    },
                    |qi, st| {
                        let t = Instant::now();
                        let r = doc.search(DenseVectorView::new(q(qi)), args.k, &cfg);
                        st.beam_ns = t.elapsed().as_nanos() as u64;
                        st.ef_used = ef;
                        r.into_iter().map(|sv| sv.vector).collect()
                    },
                );
            }
        }
        Mode::Qlr => {
            let router = router.as_ref().expect("--router is required for --mode qlr");
            let epq = args.ep_per_query.unwrap_or(router.ep.k_ep);
            let mut scratch = Vec::new();
            // The configurations: the cross product of the lists, or exactly those of the file.
            // In cache mode ef only drives the fallback search of unrouted queries.
            let mut cfgs: Vec<(usize, f32, bool, usize, usize)> = Vec::new(); // k', th, adaptive, ef_min, ef
            if let Some(path) = &args.configs_file {
                for r in read_config_rows(path) {
                    let get = |k: &str| r.get(k).map(String::as_str).unwrap_or("");
                    cfgs.push((
                        get("k_prime").parse().expect("k_prime"),
                        get("th").parse().expect("th"),
                        !matches!(get("adaptive"), "false" | "0"),
                        get("ef_min").parse().unwrap_or(kannolo::qlr::EF_MIN),
                        get("ef").parse().expect("ef"),
                    ));
                }
            } else {
                for &kp in &args.kprime_list {
                    for &th in &args.th_list {
                        for &adaptive in &adaptive_list {
                            for &ef_min in &args.ef_min_list {
                                for &ef in &args.ef_list {
                                    cfgs.push((kp, th, adaptive, ef_min, ef));
                                }
                            }
                        }
                    }
                }
            }
            for (kp, th, adaptive, ef_min, ef) in cfgs {
                // ef_min = 10 keeps the config names of the runs made before it was a knob.
                let efm = if ef_min == kannolo::qlr::EF_MIN { String::new() } else { format!("_efmin{ef_min}") };
                let p = QlrParams {
                    k_prime: kp,
                    ep_per_query: epq,
                    ef,
                    th,
                    adaptive,
                    ef_min,
                    reuse_seed_scores: args.reuse_seed_scores,
                    action,
                };
                runner.run(
                    Row {
                        config: format!("qlr_kp{kp}_epq{epq}_th{th}_ef{ef}{efm}_ad{}_{action_s}", adaptive as u8),
                        k_prime: kp.to_string(),
                        ep_per_query: epq.to_string(),
                        th: th.to_string(),
                        ef,
                        adaptive: adaptive.to_string(),
                        action: action_s.clone(),
                        w: String::new(),
                        ef_min,
                    },
                    |qi, st| {
                        router
                            .search(&doc, q(qi), args.k, &p, &mut scratch, st)
                            .into_iter()
                            .map(|sv| sv.vector)
                            .collect()
                    },
                );
            }
        }
        Mode::Window => {
            let router = router.as_ref().expect("--router is required (projection, Q3)");
            let stream_ds = read_npy_f32::<DotProduct>(args.stream.as_ref().expect("--stream"))
                .expect("reading stream");
            let stream: Vec<f32> = vectorium::DenseDataset::values(&stream_ds).to_vec();
            drop(stream_ds);
            let pos = read_i64(args.test_stream_pos.as_ref().expect("--test-stream-pos"));
            assert_eq!(pos.len(), nq);
            let ns = stream.len() / dim;
            println!("[qlr_search] building window EP for {ns} stream queries...");
            let ep = EpTable::build(&doc, &stream, dim, args.window_k_ep, 200);
            // Projected stream: the router state a deployed system would hold for past queries.
            let (proj_stream, pdim) = match &router.projection {
                Some(p) => {
                    let mut out = Vec::with_capacity(ns * p.out_dim);
                    let mut y = Vec::new();
                    for r in 0..ns {
                        p.project_into(&stream[r * dim..(r + 1) * dim], &mut y);
                        out.extend_from_slice(&y);
                    }
                    (out, p.out_dim)
                }
                None => (stream.clone(), dim),
            };
            let epq = args.ep_per_query.unwrap_or(args.window_k_ep);
            let mut scratch = Vec::new();
            for &w in &args.w_list {
                for &kp in &args.kprime_list {
                    for &th in &args.th_list {
                        for &adaptive in &adaptive_list {
                            for &ef in &args.ef_list {
                                let p = QlrParams {
                                    k_prime: kp,
                                    ep_per_query: epq,
                                    ef,
                                    th,
                                    adaptive,
                                    ef_min: args.ef_min_list[0],
                                    reuse_seed_scores: args.reuse_seed_scores,
                                    action,
                                };
                                runner.run(
                                    Row {
                                        config: format!(
                                            "win_w{w}_kp{kp}_th{th}_ef{ef}_ad{}",
                                            adaptive as u8
                                        ),
                                        k_prime: kp.to_string(),
                                        ep_per_query: epq.to_string(),
                                        th: th.to_string(),
                                        ef,
                                        adaptive: adaptive.to_string(),
                                        action: action_s.clone(),
                                        w: w.to_string(),
                                        ef_min: args.ef_min_list[0],
                                    },
                                    |qi, st| {
                                        let query = q(qi);
                                        let t = Instant::now();
                                        let short: &[f32] = match &router.projection {
                                            Some(pr) => {
                                                pr.project_into(query, &mut scratch);
                                                &scratch
                                            }
                                            None => query,
                                        };
                                        st.proj_ns = t.elapsed().as_nanos() as u64;
                                        let t = Instant::now();
                                        let hi = pos[qi];
                                        let lo = hi.saturating_sub(w);
                                        let hits = window_top(&proj_stream, pdim, lo, hi, short, kp);
                                        st.iq_ns = t.elapsed().as_nanos() as u64;
                                        route_from_hits(
                                            &doc, &ep, router.q3_score, query, &hits, args.k, &p, st,
                                        )
                                        .into_iter()
                                        .map(|sv| sv.vector)
                                        .collect()
                                    },
                                );
                            }
                        }
                    }
                }
            }
        }
        Mode::Catapult => {
            // Log warm-up material: (hash input, best result) per log row, in log order.
            let log: Option<(Vec<f32>, Vec<u64>)> = args.log_file.as_ref().map(|lf| {
                let router = router.as_ref().expect("--router is required for the log warm-up");
                let ds = read_npy_f32::<DotProduct>(lf).expect("reading log");
                let v: Vec<f32> = vectorium::DenseDataset::values(&ds).to_vec();
                let n = v.len() / dim;
                assert_eq!(n, router.ep.len(), "log rows must match the router's EP table");
                let best = (0..n).map(|i| router.ep.entries(i, 1).next().expect("empty EP row") as u64).collect();
                (v, best)
            });
            // The configurations: the cross product of the lists, or exactly those of the file.
            let mut cfgs: Vec<(usize, usize, String, String, usize)> = Vec::new(); // bits, b, warm-up, entry, ef
            if let Some(path) = &args.configs_file {
                for r in read_config_rows(path) {
                    let get = |k: &str| r.get(k).cloned().unwrap_or_default();
                    cfgs.push((
                        get("lsh_bits").parse().expect("lsh_bits"),
                        get("bucket_size").parse().expect("bucket_size"),
                        get("warmup"),
                        get("entry"),
                        get("ef").parse().expect("ef"),
                    ));
                }
            } else {
                for &bits in &args.lsh_bits_list {
                    for &cap in &args.bucket_size_list {
                        for warm in &args.catapult_warmup_list {
                            for entry_mode in &args.catapult_entry_list {
                                for &ef in &args.ef_list {
                                    cfgs.push((bits, cap, warm.clone(), entry_mode.clone(), ef));
                                }
                            }
                        }
                    }
                }
            }
            // Hash codes of the log queries, per number of hyperplanes (computed when first needed).
            let mut log_hashes: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
            for (bits, cap, warm, entry_mode, ef) in cfgs {
                let descent = match entry_mode.as_str() {
                    "descent" => true,
                    "fallback" => false,
                    m => panic!("unknown catapult entry mode {m}"),
                };
                let mut cat = Catapults::new(dim, bits, cap, args.lsh_seed);
                match warm.as_str() {
                    "none" => {}
                    "log" => {
                        let (v, best) = log.as_ref().expect("--log-file is required for the log warm-up");
                        let hashes = log_hashes.entry(bits).or_insert_with(|| {
                            let c = Catapults::new(dim, bits, 1, args.lsh_seed);
                            v.chunks_exact(dim).map(|x| c.hash(x)).collect()
                        });
                        for (&h, &d) in hashes.iter().zip(best) {
                            cat.insert(h, d);
                        }
                    }
                    w => panic!("unknown warm-up {w}"),
                }
                let cfg = HNSWSearchConfiguration::default().with_ef_search(ef);
                runner.run(
                    Row {
                        config: format!("cat_L{bits}_b{cap}_warm{warm}_{entry_mode}_ef{ef}"),
                        k_prime: bits.to_string(),
                        ep_per_query: cap.to_string(),
                        th: warm.clone(),
                        ef,
                        adaptive: String::new(),
                        action: entry_mode.clone(),
                        w: String::new(),
                        ef_min: 0,
                    },
                    |qi, st| {
                        let query = q(qi);
                        let qv = DenseVectorView::new(query);
                        let t = Instant::now();
                        let h = cat.hash(query);
                        st.proj_ns = t.elapsed().as_nanos() as u64;
                        // Seeds: the bucket's catapults, plus HNSW's default entry
                        // (always with `descent`, only for empty buckets with `fallback`).
                        let t = Instant::now();
                        let ids: Vec<u64> = cat.bucket(h).iter().copied().collect();
                        let mut seeds = doc.score_ids(qv, &ids, &());
                        if descent || seeds.is_empty() {
                            let entry = doc.ground_entry(qv, &());
                            if !seeds.iter().any(|sv| sv.vector == entry.vector) {
                                seeds.push(entry);
                            }
                        }
                        st.score_ns = t.elapsed().as_nanos() as u64;
                        st.routed = !ids.is_empty();
                        st.n_seeds = seeds.len();
                        st.ef_used = ef;
                        let t = Instant::now();
                        let res = doc.search_from_scored_seeds(qv, &seeds, args.k, &cfg);
                        st.beam_ns = t.elapsed().as_nanos() as u64;
                        // Online update, inside the timed region.
                        if let Some(best) = res.first() {
                            cat.insert(h, best.vector);
                        }
                        res.into_iter().map(|sv| sv.vector).collect()
                    },
                );
                println!("[qlr_search]   occupied buckets {} / {}", cat.occupied(), 1usize << bits);
            }
        }
    }
}
