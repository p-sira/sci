use sci_interpolate::{SurfaceSpline, fitting::make_interp_spline, multidimensional, spline};

#[test]
fn multilinear_grid_skips_singletons_without_changing_affine_values() {
    let axes = vec![
        vec![0.0, 1.0],
        vec![3.0],
        vec![4.0, 2.0, 0.0],
        vec![-1.0],
        vec![0.0, 2.0],
        vec![7.0],
        vec![8.0],
        vec![9.0],
    ];
    let field = |a: f64, b: f64, c: f64| 1.0 + 2.0 * a + 3.0 * b - c;
    let mut values = Vec::new();
    for &a in &axes[0] {
        for &b in &axes[2] {
            for &c in &axes[4] {
                values.push(field(a, b, c));
            }
        }
    }
    let queries = vec![
        vec![0.2, 3.0, 3.5, -1.0, 0.5, 7.0, 8.0, 9.0],
        vec![1.0, 3.0, 0.0, -1.0, 2.0, 7.0, 8.0, 9.0],
        vec![0.0, 3.0, 4.0, -1.0, 0.0, 7.0, 8.0, 9.0],
    ];
    let result = multidimensional::interpn(
        &axes,
        &values,
        &queries,
        multidimensional::GridMethod::Linear,
        true,
        None,
    )
    .unwrap();
    for (q, &value) in queries.iter().zip(&result) {
        assert!((value - field(q[0], q[2], q[4])).abs() < 1e-12);
    }
    let outside = [vec![0.5, 4.0, 2.0, -1.0, 1.0, 7.0, 8.0, 9.0]];
    assert_eq!(
        multidimensional::interpn(
            &axes,
            &values,
            &outside,
            multidimensional::GridMethod::Linear,
            false,
            Some(-17.0),
        )
        .unwrap(),
        vec![-17.0]
    );
    assert!(
        multidimensional::interpn(
            &axes,
            &values,
            &outside,
            multidimensional::GridMethod::Linear,
            true,
            None,
        )
        .is_err()
    );
}

#[test]
fn cubic_spline_derivatives_reproduce_a_known_polynomial() {
    let x: Vec<f64> = (0..8).map(f64::from).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&x| x.powi(3) - 2.0 * x * x + 3.0 * x + 4.0)
        .collect();
    let interpolant = make_interp_spline(&x, &y, 3).unwrap();
    let queries = [0.0, 0.25, 1.0, 3.5, 5.5, 7.0];
    let derivatives = spline::spalde(&queries, &interpolant).unwrap();
    for (&x, row) in queries.iter().zip(&derivatives) {
        let expected = [
            x.powi(3) - 2.0 * x * x + 3.0 * x + 4.0,
            3.0 * x * x - 4.0 * x + 3.0,
            6.0 * x - 4.0,
            6.0,
        ];
        for (&value, &target) in row.iter().zip(&expected) {
            assert!((value - target).abs() < 1e-9, "x={x}: {value} != {target}");
        }
    }
    assert_eq!(
        spline::spalde(&[], &interpolant).unwrap(),
        Vec::<Vec<f64>>::new()
    );
    assert!(spline::spalde(&[-0.01], &interpolant).is_err());
}

#[test]
fn scattered_triangles_preserve_affine_fields_and_fill_outside_hull() {
    let points: Vec<Vec<f64>> = (0..12)
        .flat_map(|i| (0..12).map(move |j| vec![-1.0 + i as f64 / 5.5, -1.0 + j as f64 / 5.5]))
        .collect();
    let field = |x: f64, y: f64| 3.0 + 1.5 * x - 2.0 * y;
    let values: Vec<f64> = points.iter().map(|p| field(p[0], p[1])).collect();
    let mut queries: Vec<Vec<f64>> = (0..11)
        .flat_map(|i| (0..10).map(move |j| vec![-0.9 + i as f64 * 0.18, -0.9 + j as f64 * 0.2]))
        .collect();
    queries.push(vec![2.0, 0.0]);
    queries.push(vec![0.0, -2.0]);
    for rescale in [false, true] {
        let actual = multidimensional::griddata(
            &points,
            &values,
            &queries,
            multidimensional::ScatteredMethod::Linear,
            -17.0,
            rescale,
        )
        .unwrap();
        for (q, &value) in queries[..queries.len() - 2]
            .iter()
            .zip(&actual[..actual.len() - 2])
        {
            assert!((value - field(q[0], q[1])).abs() < 1e-12, "{q:?}: {value}");
        }
        assert_eq!(&actual[actual.len() - 2..], &[-17.0, -17.0]);
    }
    assert!(
        multidimensional::griddata(
            &points,
            &values,
            &[vec![f64::NAN, 0.0]],
            multidimensional::ScatteredMethod::Linear,
            -17.0,
            false,
        )
        .is_err()
    );
}

#[test]
fn tensor_surface_cached_bases_preserve_polynomials_and_clamping() {
    let plane = SurfaceSpline {
        x_knots: vec![0.0, 0.0, 1.0, 1.0],
        y_knots: vec![0.0, 0.0, 1.0, 1.0],
        coefficients: vec![3.0, 5.0, 4.0, 6.0],
        x_degree: 1,
        y_degree: 1,
    };
    let xs = [-1.0, 0.0, 0.3, 1.0, 2.0];
    let ys = [-2.0, 0.0, 0.4, 1.0, 3.0];
    let result = multidimensional::bisplev(&xs, &ys, &plane, 0, 0).unwrap();
    for (&x, row) in xs.iter().zip(&result) {
        for (&y, &value) in ys.iter().zip(row) {
            let expected = 3.0 + x.clamp(0.0, 1.0) + 2.0 * y.clamp(0.0, 1.0);
            assert!((value - expected).abs() < 1e-14);
        }
    }
    assert_eq!(
        multidimensional::bisplev(&[], &ys, &plane, 0, 0)
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        multidimensional::bisplev(&xs, &[], &plane, 0, 0).unwrap(),
        vec![vec![]; xs.len()]
    );

    let cubics = [0.0, 0.0, 0.0, 1.0];
    let surface = SurfaceSpline {
        x_knots: vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        y_knots: vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        coefficients: (0..4)
            .flat_map(|i| (0..4).map(move |j| cubics[i] + 2.0 * cubics[j]))
            .collect(),
        x_degree: 3,
        y_degree: 3,
    };
    let xs = [0.0, 0.125, 0.4, 0.8, 1.0];
    let ys = [0.0, 0.25, 0.5, 1.0];
    for (&x, row) in xs
        .iter()
        .zip(multidimensional::bisplev(&xs, &ys, &surface, 0, 0).unwrap())
    {
        for (&y, value) in ys.iter().zip(row) {
            assert!((value - (x.powi(3) + 2.0 * y.powi(3))).abs() < 1e-13);
        }
    }
}
