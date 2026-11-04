use nalgebra::{DMatrix, DVector};
use sci_spatial::{Error, Result, transform::tsearch};

// Independent reference retaining the former allocating nalgebra multiplication.
// These inputs are validated by the fixtures below, so only the geometric
// calculation is reproduced here.
fn reference_tsearch(
    vertices: &[Vec<f64>],
    simplices: &[Vec<usize>],
    queries: &[Vec<f64>],
    tol: f64,
) -> Result<Vec<Option<usize>>> {
    let d = vertices[0].len();
    let cells = simplices
        .iter()
        .map(|cell| {
            let origin = &vertices[cell[0]];
            let edges = DMatrix::from_fn(d, d, |i, j| vertices[cell[j + 1]][i] - origin[i]);
            let inverse = edges
                .lu()
                .try_inverse()
                .ok_or(Error::NumericalFailure("degenerate simplex"))?;
            Ok((origin, inverse))
        })
        .collect::<Result<Vec<_>>>()?;

    queries
        .iter()
        .map(|q| {
            for (i, (origin, inverse)) in cells.iter().enumerate() {
                let delta = DVector::from_iterator(d, q.iter().zip(*origin).map(|(&x, &a)| x - a));
                let bary = inverse * delta;
                if bary.iter().any(|x| !x.is_finite()) {
                    return Err(Error::NonFiniteEvaluation);
                }
                let last = 1.0 - bary.iter().sum::<f64>();
                if !last.is_finite() {
                    return Err(Error::NonFiniteEvaluation);
                }
                if last >= -tol
                    && last <= 1.0 + tol
                    && bary.iter().all(|&x| x >= -tol && x <= 1.0 + tol)
                {
                    return Ok(Some(i));
                }
            }
            Ok(None)
        })
        .collect()
}

fn simplex(d: usize) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
    let mut vertices = vec![vec![0.0; d]];
    for axis in 0..d {
        let mut corner = vec![0.0; d];
        corner[axis] = 1.0;
        vertices.push(corner);
    }
    (vertices, vec![(0..=d).collect()])
}

#[test]
fn simplex_fixtures_cover_dimensions_one_two_three_and_sixty_four() {
    for d in [1, 2, 3, 64] {
        let (vertices, cells) = simplex(d);
        let mut on_face = vec![0.0; d];
        if d == 1 {
            on_face[0] = 1.0;
        } else {
            on_face[0] = 0.5;
            on_face[1] = 0.5;
        }
        let mut outside = vec![0.0; d];
        outside[0] = 1.25;
        let mut slightly_outside = vec![0.0; d];
        slightly_outside[0] = -5e-13;
        let mut beyond_tolerance = vec![0.0; d];
        beyond_tolerance[0] = -2e-12;
        let queries = vec![
            vec![0.0; d],
            vec![0.5 / d as f64; d],
            on_face,
            outside,
            slightly_outside,
            beyond_tolerance,
        ];

        let without_tolerance = tsearch(&vertices, &cells, &queries, 0.0).unwrap();
        assert_eq!(
            without_tolerance,
            vec![Some(0), Some(0), Some(0), None, None, None],
            "dimension {d}, tolerance 0"
        );
        let with_tolerance = tsearch(&vertices, &cells, &queries, 1e-12).unwrap();
        assert_eq!(
            with_tolerance,
            vec![Some(0), Some(0), Some(0), None, Some(0), None],
            "dimension {d}, tolerance 1e-12"
        );
        for tol in [0.0, 1e-12] {
            assert_eq!(
                tsearch(&vertices, &cells, &queries, tol),
                reference_tsearch(&vertices, &cells, &queries, tol),
                "reference mismatch in {d}-D at tol={tol}"
            );
        }
    }
}

fn affine_simplex(d: usize) -> Vec<Vec<f64>> {
    let origin = vec![0.25; d];
    let mut vertices = vec![origin.clone()];
    for axis in 0..d {
        let mut vertex = origin.clone();
        vertex[axis] += 1.0;
        if axis + 1 < d {
            vertex[axis + 1] += 0.125;
        }
        vertices.push(vertex);
    }
    vertices
}

fn interpolate(vertices: &[Vec<f64>], bary: &[f64]) -> Vec<f64> {
    let mut q = vertices[0].clone();
    for (axis, &weight) in bary.iter().enumerate() {
        for (j, qj) in q.iter_mut().enumerate() {
            *qj += weight * (vertices[axis + 1][j] - vertices[0][j]);
        }
    }
    q
}

#[test]
fn affine_simplices_agree_with_allocating_nalgebra_reference() {
    for d in [1, 2, 3, 64] {
        let vertices = affine_simplex(d);
        let cells = vec![(0..=d).collect()];
        let mut outside = vec![0.0; d];
        outside[0] = 1.5;
        let inside = vec![0.25 / d as f64; d];
        let mut queries = vec![
            interpolate(&vertices, &inside),
            interpolate(&vertices, &outside),
            vertices[0].clone(),
        ];
        queries.extend(vertices.iter().skip(1).cloned());
        let expected = reference_tsearch(&vertices, &cells, &queries, 1e-12);
        assert_eq!(
            tsearch(&vertices, &cells, &queries, 1e-12),
            expected,
            "affine {d}-D"
        );
        assert_eq!(
            tsearch(&vertices, &cells, &queries[..2], 1e-12).unwrap(),
            vec![Some(0), None],
            "analytical affine {d}-D"
        );
    }
}

#[test]
fn overlapping_faces_choose_the_first_simplex_with_zero_tolerance() {
    let vertices = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let cells = vec![vec![0, 1, 2], vec![1, 3, 2], vec![0, 1, 2]];
    let queries = vec![
        vec![0.5, 0.5],
        vec![0.25, 0.25],
        vec![0.75, 0.75],
        vec![1.5, 1.5],
        vec![1.0, 0.0],
    ];
    let expected = vec![Some(0), Some(0), Some(1), None, Some(0)];
    assert_eq!(tsearch(&vertices, &cells, &queries, 0.0), Ok(expected));
    assert_eq!(
        tsearch(&vertices, &cells, &queries, 0.0),
        reference_tsearch(&vertices, &cells, &queries, 0.0)
    );
}

#[test]
fn near_degenerate_faces_match_the_original_nalgebra_path() {
    let vertices = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![1.0, 1e-12]];
    let cells = vec![vec![0, 1, 2]];
    let queries = vec![
        vec![0.5, 2.5e-13],
        vec![0.25, -5e-25],
        vec![0.5, 5e-13],
        vec![0.5, 6e-13],
        vec![0.0, 0.0],
        vec![1.0, 0.0],
    ];
    for tol in [0.0, 1e-12, 1e-9] {
        assert_eq!(
            tsearch(&vertices, &cells, &queries, tol),
            reference_tsearch(&vertices, &cells, &queries, tol),
            "near-degenerate simplex at tol={tol}"
        );
    }
    assert_eq!(
        tsearch(&vertices, &cells, &queries[..1], 0.0),
        Ok(vec![Some(0)])
    );
}

#[test]
fn invalid_inputs_and_arithmetic_overflow_keep_the_same_errors() {
    let triangle = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
    assert_eq!(
        tsearch(&triangle, &[vec![0, 1]], &[], 0.0),
        Err(Error::DimensionMismatch)
    );
    assert_eq!(
        tsearch(&triangle, &[vec![0, 1, 3]], &[], 0.0),
        Err(Error::InvalidInput("simplex vertex index out of bounds"))
    );
    assert_eq!(
        tsearch(&triangle, &[vec![0, 0, 1]], &[], 0.0),
        Err(Error::NumericalFailure("degenerate simplex"))
    );
    assert_eq!(
        tsearch(&triangle, &[vec![0, 1, 2]], &[vec![0.0]], 0.0),
        Err(Error::DimensionMismatch)
    );
    assert_eq!(
        tsearch(&triangle, &[vec![0, 1, 2]], &[], -1.0),
        Err(Error::InvalidInput(
            "invalid simplex dimension or tolerance"
        ))
    );
    assert_eq!(
        tsearch(&triangle, &[vec![0, 1, 2]], &[], f64::NAN),
        Err(Error::InvalidInput(
            "invalid simplex dimension or tolerance"
        ))
    );
    assert_eq!(
        tsearch(&[vec![0.0; 65]], &[], &[], 0.0),
        Err(Error::InvalidInput(
            "invalid simplex dimension or tolerance"
        ))
    );
    assert!(tsearch(&triangle, &[vec![0, 1, 2]], &[vec![f64::NAN, 0.0]], 0.0).is_err());

    // Input validation is required even when no query can be classified.
    assert_eq!(
        tsearch(&[vec![f64::MAX], vec![-f64::MAX]], &[vec![0, 1]], &[], 0.0),
        Err(Error::NonFiniteEvaluation)
    );
    // A finite inverse may still multiply an overflowing query-origin delta.
    assert_eq!(
        tsearch(
            &[vec![1e308], vec![0.0]],
            &[vec![0, 1]],
            &[vec![-1e308]],
            0.0
        ),
        Err(Error::NonFiniteEvaluation)
    );

    let many_cells = vec![vec![0, 1, 2]; 4_000];
    let many_queries = vec![vec![0.0, 0.0]; 1_000];
    assert_eq!(
        tsearch(&triangle, &many_cells, &many_queries, 0.0),
        Err(Error::InvalidInput("geometry work limit exceeded"))
    );
}
