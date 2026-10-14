# scipy.spatial

[Back to SciPy function index](../SCIPY_FUNCTIONS.md)

## Namespaces

- [`scipy.spatial`](#scipy-spatial)
- [`scipy.spatial.ckdtree`](#scipy-spatial-ckdtree) (additional / compatibility)
- [`scipy.spatial.distance`](#scipy-spatial-distance)
- [`scipy.spatial.kdtree`](#scipy-spatial-kdtree) (additional / compatibility)
- [`scipy.spatial.qhull`](#scipy-spatial-qhull) (additional / compatibility)
- [`scipy.spatial.transform`](#scipy-spatial-transform)
- [`scipy.spatial.transform.rotation`](#scipy-spatial-transform-rotation) (additional / compatibility)

## Public API namespaces

### scipy.spatial

[Module reference](https://docs.scipy.org/doc/scipy/reference/spatial.html#module-scipy.spatial)

Function entries: **9**. Callable objects: **0**. Classes: **9**.

#### Functions

- [`convex_hull_plot_2d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.convex_hull_plot_2d.html#scipy.spatial.convex_hull_plot_2d)
- [`delaunay_plot_2d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.delaunay_plot_2d.html#scipy.spatial.delaunay_plot_2d)
- [`distance_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance_matrix.html#scipy.spatial.distance_matrix)
- [`geometric_slerp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.geometric_slerp.html#scipy.spatial.geometric_slerp)
- [`minkowski_distance`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.minkowski_distance.html#scipy.spatial.minkowski_distance)
- [`minkowski_distance_p`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.minkowski_distance_p.html#scipy.spatial.minkowski_distance_p)
- [`procrustes`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html#scipy.spatial.procrustes)
- [`tsearch`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.tsearch.html#scipy.spatial.tsearch)
- [`voronoi_plot_2d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.voronoi_plot_2d.html#scipy.spatial.voronoi_plot_2d)

#### Classes and their public methods

- [`ConvexHull`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html#scipy.spatial.convexhull)
  - [`ConvexHull.add_points`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.add_points.html#scipy.spatial.convexhull.add_points)
  - [`ConvexHull.close`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.close.html#scipy.spatial.convexhull.close)
- [`Delaunay`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html#scipy.spatial.delaunay)
  - [`Delaunay.add_points`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.add_points.html#scipy.spatial.delaunay.add_points)
  - [`Delaunay.close`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.close.html#scipy.spatial.delaunay.close)
  - [`Delaunay.find_simplex`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.find_simplex.html#scipy.spatial.delaunay.find_simplex)
  - [`Delaunay.lift_points`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.lift_points.html#scipy.spatial.delaunay.lift_points)
  - [`Delaunay.plane_distance`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.plane_distance.html#scipy.spatial.delaunay.plane_distance)
- [`HalfspaceIntersection`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.HalfspaceIntersection.html#scipy.spatial.halfspaceintersection)
  - [`HalfspaceIntersection.add_halfspaces`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.HalfspaceIntersection.add_halfspaces.html#scipy.spatial.halfspaceintersection.add_halfspaces)
  - [`HalfspaceIntersection.close`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.HalfspaceIntersection.close.html#scipy.spatial.halfspaceintersection.close)
- [`KDTree`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html#scipy.spatial.kdtree)
  - [`KDTree.count_neighbors`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.count_neighbors.html#scipy.spatial.kdtree.count_neighbors)
  - [`KDTree.query`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html#scipy.spatial.kdtree.query)
  - [`KDTree.query_ball_point`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query_ball_point.html#scipy.spatial.kdtree.query_ball_point)
  - [`KDTree.query_ball_tree`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query_ball_tree.html#scipy.spatial.kdtree.query_ball_tree)
  - [`KDTree.query_pairs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query_pairs.html#scipy.spatial.kdtree.query_pairs)
  - [`KDTree.sparse_distance_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.sparse_distance_matrix.html#scipy.spatial.kdtree.sparse_distance_matrix)
- [`QhullError`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.QhullError.html#scipy.spatial.qhullerror)
  - [`QhullError.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.QhullError.html#scipy.spatial.qhullerror.add_note)
  - [`QhullError.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.QhullError.html#scipy.spatial.qhullerror.with_traceback)
- [`Rectangle`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Rectangle.html#scipy.spatial.rectangle)
  - [`Rectangle.max_distance_point`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Rectangle.max_distance_point.html#scipy.spatial.rectangle.max_distance_point)
  - [`Rectangle.max_distance_rectangle`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Rectangle.max_distance_rectangle.html#scipy.spatial.rectangle.max_distance_rectangle)
  - [`Rectangle.min_distance_point`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Rectangle.min_distance_point.html#scipy.spatial.rectangle.min_distance_point)
  - [`Rectangle.min_distance_rectangle`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Rectangle.min_distance_rectangle.html#scipy.spatial.rectangle.min_distance_rectangle)
  - [`Rectangle.split`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Rectangle.split.html#scipy.spatial.rectangle.split)
  - [`Rectangle.volume`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Rectangle.volume.html#scipy.spatial.rectangle.volume)
- [`SphericalVoronoi`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.SphericalVoronoi.html#scipy.spatial.sphericalvoronoi)
  - [`SphericalVoronoi.calculate_areas`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.SphericalVoronoi.calculate_areas.html#scipy.spatial.sphericalvoronoi.calculate_areas)
  - [`SphericalVoronoi.sort_vertices_of_regions`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.SphericalVoronoi.sort_vertices_of_regions.html#scipy.spatial.sphericalvoronoi.sort_vertices_of_regions)
- [`Voronoi`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html#scipy.spatial.voronoi)
  - [`Voronoi.add_points`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.add_points.html#scipy.spatial.voronoi.add_points)
  - [`Voronoi.close`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.close.html#scipy.spatial.voronoi.close)
- [`cKDTree`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html#scipy.spatial.ckdtree)
  - [`cKDTree.count_neighbors`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.count_neighbors.html#scipy.spatial.ckdtree.count_neighbors)
  - [`cKDTree.query`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html#scipy.spatial.ckdtree.query)
  - [`cKDTree.query_ball_point`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_ball_point.html#scipy.spatial.ckdtree.query_ball_point)
  - [`cKDTree.query_ball_tree`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_ball_tree.html#scipy.spatial.ckdtree.query_ball_tree)
  - [`cKDTree.query_pairs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_pairs.html#scipy.spatial.ckdtree.query_pairs)
  - [`cKDTree.sparse_distance_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.sparse_distance_matrix.html#scipy.spatial.ckdtree.sparse_distance_matrix)

### scipy.spatial.distance

[Module reference](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html#module-scipy.spatial.distance)

Function entries: **27**. Callable objects: **0**. Classes: **0**.

#### Functions

- [`braycurtis`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.braycurtis.html#scipy.spatial.distance.braycurtis)
- [`canberra`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.canberra.html#scipy.spatial.distance.canberra)
- [`cdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist)
- [`chebyshev`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.chebyshev.html#scipy.spatial.distance.chebyshev)
- [`cityblock`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cityblock.html#scipy.spatial.distance.cityblock)
- [`correlation`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.correlation.html#scipy.spatial.distance.correlation)
- [`cosine`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html#scipy.spatial.distance.cosine)
- [`dice`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.dice.html#scipy.spatial.distance.dice)
- [`directed_hausdorff`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html#scipy.spatial.distance.directed_hausdorff)
- [`euclidean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html#scipy.spatial.distance.euclidean)
- [`hamming`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.hamming.html#scipy.spatial.distance.hamming)
- [`is_valid_dm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.is_valid_dm.html#scipy.spatial.distance.is_valid_dm)
- [`is_valid_y`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.is_valid_y.html#scipy.spatial.distance.is_valid_y)
- [`jaccard`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html#scipy.spatial.distance.jaccard)
- [`jensenshannon`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html#scipy.spatial.distance.jensenshannon)
- [`mahalanobis`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.mahalanobis.html#scipy.spatial.distance.mahalanobis)
- [`minkowski`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.minkowski.html#scipy.spatial.distance.minkowski)
- [`num_obs_dm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.num_obs_dm.html#scipy.spatial.distance.num_obs_dm)
- [`num_obs_y`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.num_obs_y.html#scipy.spatial.distance.num_obs_y)
- [`pdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist)
- [`rogerstanimoto`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.rogerstanimoto.html#scipy.spatial.distance.rogerstanimoto)
- [`russellrao`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.russellrao.html#scipy.spatial.distance.russellrao)
- [`seuclidean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.seuclidean.html#scipy.spatial.distance.seuclidean)
- [`sokalsneath`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.sokalsneath.html#scipy.spatial.distance.sokalsneath)
- [`sqeuclidean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.sqeuclidean.html#scipy.spatial.distance.sqeuclidean)
- [`squareform`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html#scipy.spatial.distance.squareform)
- [`yule`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.yule.html#scipy.spatial.distance.yule)

### scipy.spatial.transform

[Module reference](https://docs.scipy.org/doc/scipy/reference/spatial.transform.html#module-scipy.spatial.transform)

Function entries: **0**. Callable objects: **0**. Classes: **4**.

#### Classes and their public methods

- [`RigidTransform`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RigidTransform.html#scipy.spatial.transform.rigidtransform)
  - [`RigidTransform.apply`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RigidTransform.apply.html#scipy.spatial.transform.rigidtransform.apply)
  - [`RigidTransform.as_components`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RigidTransform.as_components.html#scipy.spatial.transform.rigidtransform.as_components)
  - [`RigidTransform.as_dual_quat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RigidTransform.as_dual_quat.html#scipy.spatial.transform.rigidtransform.as_dual_quat)
  - [`RigidTransform.as_exp_coords`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RigidTransform.as_exp_coords.html#scipy.spatial.transform.rigidtransform.as_exp_coords)
  - [`RigidTransform.as_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RigidTransform.as_matrix.html#scipy.spatial.transform.rigidtransform.as_matrix)
  - [`RigidTransform.concatenate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RigidTransform.concatenate.html#scipy.spatial.transform.rigidtransform.concatenate)
  - [`RigidTransform.from_components`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RigidTransform.from_components.html#scipy.spatial.transform.rigidtransform.from_components)
  - [`RigidTransform.from_dual_quat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RigidTransform.from_dual_quat.html#scipy.spatial.transform.rigidtransform.from_dual_quat)
  - [`RigidTransform.from_exp_coords`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RigidTransform.from_exp_coords.html#scipy.spatial.transform.rigidtransform.from_exp_coords)
  - [`RigidTransform.from_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RigidTransform.from_matrix.html#scipy.spatial.transform.rigidtransform.from_matrix)
  - [`RigidTransform.from_rotation`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RigidTransform.from_rotation.html#scipy.spatial.transform.rigidtransform.from_rotation)
  - [`RigidTransform.from_translation`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RigidTransform.from_translation.html#scipy.spatial.transform.rigidtransform.from_translation)
  - [`RigidTransform.identity`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RigidTransform.identity.html#scipy.spatial.transform.rigidtransform.identity)
  - [`RigidTransform.inv`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RigidTransform.inv.html#scipy.spatial.transform.rigidtransform.inv)
  - [`RigidTransform.mean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RigidTransform.mean.html#scipy.spatial.transform.rigidtransform.mean)
- [`Rotation`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html#scipy.spatial.transform.rotation)
  - [`Rotation.align_vectors`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.align_vectors.html#scipy.spatial.transform.rotation.align_vectors)
  - [`Rotation.apply`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.apply.html#scipy.spatial.transform.rotation.apply)
  - [`Rotation.approx_equal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.approx_equal.html#scipy.spatial.transform.rotation.approx_equal)
  - [`Rotation.as_davenport`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_davenport.html#scipy.spatial.transform.rotation.as_davenport)
  - [`Rotation.as_euler`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_euler.html#scipy.spatial.transform.rotation.as_euler)
  - [`Rotation.as_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_matrix.html#scipy.spatial.transform.rotation.as_matrix)
  - [`Rotation.as_mrp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_mrp.html#scipy.spatial.transform.rotation.as_mrp)
  - [`Rotation.as_quat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_quat.html#scipy.spatial.transform.rotation.as_quat)
  - [`Rotation.as_rotvec`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_rotvec.html#scipy.spatial.transform.rotation.as_rotvec)
  - [`Rotation.concatenate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.concatenate.html#scipy.spatial.transform.rotation.concatenate)
  - [`Rotation.create_group`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.create_group.html#scipy.spatial.transform.rotation.create_group)
  - [`Rotation.from_davenport`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_davenport.html#scipy.spatial.transform.rotation.from_davenport)
  - [`Rotation.from_euler`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html#scipy.spatial.transform.rotation.from_euler)
  - [`Rotation.from_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_matrix.html#scipy.spatial.transform.rotation.from_matrix)
  - [`Rotation.from_mrp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_mrp.html#scipy.spatial.transform.rotation.from_mrp)
  - [`Rotation.from_quat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html#scipy.spatial.transform.rotation.from_quat)
  - [`Rotation.from_rotvec`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_rotvec.html#scipy.spatial.transform.rotation.from_rotvec)
  - [`Rotation.identity`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.identity.html#scipy.spatial.transform.rotation.identity)
  - [`Rotation.inv`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.inv.html#scipy.spatial.transform.rotation.inv)
  - [`Rotation.magnitude`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.magnitude.html#scipy.spatial.transform.rotation.magnitude)
  - [`Rotation.mean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.mean.html#scipy.spatial.transform.rotation.mean)
  - [`Rotation.random`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.random.html#scipy.spatial.transform.rotation.random)
  - [`Rotation.reduce`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.reduce.html#scipy.spatial.transform.rotation.reduce)
- [`RotationSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RotationSpline.html#scipy.spatial.transform.rotationspline)
- [`Slerp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Slerp.html#scipy.spatial.transform.slerp)

## Additional and compatibility namespaces

Many of these paths are deprecated compatibility aliases. Class and object method lists are omitted in these sections to avoid repeating the public API inventory.

### scipy.spatial.ckdtree

Function entries: **0**. Callable objects: **0**. Classes: **1**.

#### Classes

- `cKDTree`

### scipy.spatial.kdtree

Function entries: **3**. Callable objects: **0**. Classes: **3**.

#### Functions

- `distance_matrix`
- `minkowski_distance`
- `minkowski_distance_p`

#### Classes

- `KDTree`
- `Rectangle`
- `cKDTree`

### scipy.spatial.qhull

Function entries: **1**. Callable objects: **0**. Classes: **5**.

#### Functions

- `tsearch`

#### Classes

- `ConvexHull`
- `Delaunay`
- `HalfspaceIntersection`
- `QhullError`
- `Voronoi`

### scipy.spatial.transform.rotation

Function entries: **0**. Callable objects: **0**. Classes: **2**.

#### Classes

- `Rotation`
- `Slerp`
