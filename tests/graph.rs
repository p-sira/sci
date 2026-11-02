use sci::{SparseMatrix, graph};

#[test]
fn graph_feature_can_run_without_sparse_feature() -> sci::Result<()> {
    let a = SparseMatrix {
        rows: 2,
        cols: 2,
        indptr: vec![0, 1, 1],
        indices: vec![1],
        data: vec![4.0],
    };
    let shortest = graph::dijkstra(&a, &Default::default())?;
    assert_eq!(shortest.distances[0][1], 4.0);
    assert_eq!(graph::connected_components(&a, false, false)?.count, 1);
    Ok(())
}
