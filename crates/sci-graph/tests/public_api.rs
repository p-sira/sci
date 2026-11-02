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
