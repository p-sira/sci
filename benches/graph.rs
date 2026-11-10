//! Deterministic sparse graph traversal and shortest-path workloads.

use divan::{Bencher, black_box};
use sci::{SparseMatrix, graph};

fn main() {
    divan::main();
}

fn graph_matrix(n: usize) -> SparseMatrix {
    let edges: Vec<_> = (0..n)
        .flat_map(|r| (0..6).map(move |j| (r, (r + 1 + j * 23) % n, (j % 4 + 1) as f64)))
        .collect();
    graph::csr::from_coo(n, n, &edges).unwrap()
}

#[divan::bench(args = [64, 128, 256])]
fn breadth_first(bencher: Bencher, n: usize) {
    let a = graph_matrix(n);
    bencher.bench(|| graph::breadth_first_order(black_box(&a), black_box(0), true).unwrap());
}

#[divan::bench(args = [64, 128, 256])]
fn depth_first(bencher: Bencher, n: usize) {
    let a = graph_matrix(n);
    bencher.bench(|| graph::depth_first_order(black_box(&a), black_box(0), true).unwrap());
}

#[divan::bench(args = [64, 128])]
fn dijkstra_all_sources(bencher: Bencher, n: usize) {
    let a = graph_matrix(n);
    let options = graph::PathOptions::default();
    bencher.bench(|| graph::dijkstra(black_box(&a), black_box(&options)).unwrap());
}

#[divan::bench(args = [64, 128])]
fn reconstruct_distances(bencher: Bencher, n: usize) {
    let a = graph_matrix(n);
    let predecessors = graph::dijkstra(&a, &Default::default())
        .unwrap()
        .predecessors;
    bencher.bench(|| {
        graph::construct_dist_matrix(black_box(&a), black_box(&predecessors), true, f64::INFINITY)
            .unwrap()
    });
}

#[divan::bench(args = [64, 128, 256])]
fn laplacian(bencher: Bencher, n: usize) {
    let a = graph_matrix(n);
    bencher.bench(|| graph::laplacian(black_box(&a), true, true, false).unwrap());
}

#[divan::bench(args = [64, 128, 256])]
fn minimum_spanning_tree(bencher: Bencher, n: usize) {
    let a = graph_matrix(n);
    bencher.bench(|| graph::minimum_spanning_tree(black_box(&a)).unwrap());
}
