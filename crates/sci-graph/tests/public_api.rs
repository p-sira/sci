use sci_graph::{breadth_first_tree, connected_components, dijkstra, minimum_spanning_tree};

#[test]
fn direct_graph_crate_preserves_zero_weight_edges() -> sci_graph::Result<()> {
    let a = sci_graph::csr::from_coo(3, 3, &[(0, 1, 0.0), (1, 2, 2.0)])?;
    let paths = dijkstra(&a, &Default::default())?;
    assert_eq!(paths.distances[0], vec![0.0, 0.0, 2.0]);

    let components = connected_components(&a, false, false)?;
    assert_eq!(components.count, 1);
    let tree = breadth_first_tree(&a, 0, true)?;
    assert_eq!(tree.indices, vec![1, 2]);
    assert_eq!(tree.data, vec![0.0, 2.0]);

    // As documented, the spanning-tree algorithm omits explicit zeros.
    let mst = minimum_spanning_tree(&a)?;
    assert_eq!(mst.data, [2.0]);

    // Reject malformed CSR before iterating over its storage.
    let malformed = sci_graph::SparseMatrix {
        rows: 2,
        cols: 2,
        indptr: vec![0, 2, 2],
        indices: vec![1],
        data: vec![1.0],
    };
    assert!(minimum_spanning_tree(&malformed).is_err());
    Ok(())
}

#[test]
fn traversals_and_distance_reconstruction_follow_sorted_neighbors() -> sci_graph::Result<()> {
    let a = sci_graph::csr::from_coo(4, 4, &[(0, 2, 1.), (0, 1, 2.), (1, 3, 4.), (2, 3, 1.)])?;
    let depth = sci_graph::depth_first_order(&a, 0, true)?;
    assert_eq!(depth.order, [0, 1, 3, 2]);
    assert_eq!(depth.predecessors, [None, Some(0), Some(0), Some(1)]);
    let breadth = sci_graph::breadth_first_order(&a, 0, true)?;
    assert_eq!(breadth.order, [0, 1, 2, 3]);
    assert_eq!(breadth.predecessors, [None, Some(0), Some(0), Some(1)]);

    let paths = sci_graph::dijkstra(&a, &Default::default())?;
    let reconstructed =
        sci_graph::construct_dist_matrix(&a, &paths.predecessors, true, f64::INFINITY)?;
    assert_eq!(reconstructed, paths.distances);
    assert_eq!(reconstructed[0], [0.0, 2.0, 1.0, 2.0]);
    assert_eq!(
        sci_graph::reconstruct_path(&a, &paths.predecessors[0], true)?.data,
        [2.0, 1.0, 1.0]
    );
    assert!(sci_graph::reconstruct_path(&a, &[None, Some(3), None, Some(1)], true).is_err());
    assert!(sci_graph::reconstruct_path(&a, &[None, None, Some(1), None], true).is_err());
    let cycle = sci_graph::csr::from_coo(2, 2, &[(0, 1, 1.), (1, 0, 1.)])?;
    assert!(sci_graph::reconstruct_path(&cycle, &[Some(1), Some(0)], true).is_err());
    Ok(())
}

#[test]
fn sparse_laplacian_and_mst_keep_edge_conventions() -> sci_graph::Result<()> {
    let a = sci_graph::csr::from_coo(3, 3, &[(0, 1, 2.), (1, 0, 3.), (1, 2, 4.), (2, 2, 9.)])?;
    let lap = sci_graph::laplacian(&a, false, true, true)?;
    sci_graph::csr::validate(&lap.matrix)?;
    assert_eq!(lap.diagonal, [5., 9., 4.]);
    assert_eq!(
        sci_graph::csr::to_dense(&lap.matrix)?,
        [vec![5., -5., 0.], vec![-5., 9., -4.], vec![0., -4., 4.]]
    );
    let directed_zero = sci_graph::csr::from_coo(2, 2, &[(0, 1, 0.), (1, 0, 7.)])?;
    assert_eq!(sci_graph::minimum_spanning_tree(&directed_zero)?.data, [7.]);
    Ok(())
}
