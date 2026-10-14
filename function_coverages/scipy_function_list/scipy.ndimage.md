# scipy.ndimage

[Back to SciPy function index](../SCIPY_FUNCTIONS.md)

## Namespaces

- [`scipy.ndimage`](#scipy-ndimage)
- [`scipy.ndimage.filters`](#scipy-ndimage-filters) (additional / compatibility)
- [`scipy.ndimage.fourier`](#scipy-ndimage-fourier) (additional / compatibility)
- [`scipy.ndimage.interpolation`](#scipy-ndimage-interpolation) (additional / compatibility)
- [`scipy.ndimage.measurements`](#scipy-ndimage-measurements) (additional / compatibility)
- [`scipy.ndimage.morphology`](#scipy-ndimage-morphology) (additional / compatibility)

## Public API namespaces

### scipy.ndimage

[Module reference](https://docs.scipy.org/doc/scipy/reference/ndimage.html#module-scipy.ndimage)

Function entries: **75**. Callable objects: **0**. Classes: **0**.

#### Functions

- [`affine_transform`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html#scipy.ndimage.affine_transform)
- [`binary_closing`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_closing.html#scipy.ndimage.binary_closing)
- [`binary_dilation`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_dilation.html#scipy.ndimage.binary_dilation)
- [`binary_erosion`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_erosion.html#scipy.ndimage.binary_erosion)
- [`binary_fill_holes`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_fill_holes.html#scipy.ndimage.binary_fill_holes)
- [`binary_hit_or_miss`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_hit_or_miss.html#scipy.ndimage.binary_hit_or_miss)
- [`binary_opening`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_opening.html#scipy.ndimage.binary_opening)
- [`binary_propagation`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_propagation.html#scipy.ndimage.binary_propagation)
- [`black_tophat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.black_tophat.html#scipy.ndimage.black_tophat)
- [`center_of_mass`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.center_of_mass.html#scipy.ndimage.center_of_mass)
- [`convolve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html#scipy.ndimage.convolve)
- [`convolve1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve1d.html#scipy.ndimage.convolve1d)
- [`correlate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.correlate.html#scipy.ndimage.correlate)
- [`correlate1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.correlate1d.html#scipy.ndimage.correlate1d)
- [`distance_transform_bf`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_bf.html#scipy.ndimage.distance_transform_bf)
- [`distance_transform_cdt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_cdt.html#scipy.ndimage.distance_transform_cdt)
- [`distance_transform_edt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html#scipy.ndimage.distance_transform_edt)
- [`extrema`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.extrema.html#scipy.ndimage.extrema)
- [`find_objects`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.find_objects.html#scipy.ndimage.find_objects)
- [`fourier_ellipsoid`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.fourier_ellipsoid.html#scipy.ndimage.fourier_ellipsoid)
- [`fourier_gaussian`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.fourier_gaussian.html#scipy.ndimage.fourier_gaussian)
- [`fourier_shift`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.fourier_shift.html#scipy.ndimage.fourier_shift)
- [`fourier_uniform`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.fourier_uniform.html#scipy.ndimage.fourier_uniform)
- [`gaussian_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html#scipy.ndimage.gaussian_filter)
- [`gaussian_filter1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html#scipy.ndimage.gaussian_filter1d)
- [`gaussian_gradient_magnitude`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_gradient_magnitude.html#scipy.ndimage.gaussian_gradient_magnitude)
- [`gaussian_laplace`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_laplace.html#scipy.ndimage.gaussian_laplace)
- [`generate_binary_structure`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generate_binary_structure.html#scipy.ndimage.generate_binary_structure)
- [`generic_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generic_filter.html#scipy.ndimage.generic_filter)
- [`generic_filter1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generic_filter1d.html#scipy.ndimage.generic_filter1d)
- [`generic_gradient_magnitude`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generic_gradient_magnitude.html#scipy.ndimage.generic_gradient_magnitude)
- [`generic_laplace`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generic_laplace.html#scipy.ndimage.generic_laplace)
- [`geometric_transform`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.geometric_transform.html#scipy.ndimage.geometric_transform)
- [`grey_closing`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_closing.html#scipy.ndimage.grey_closing)
- [`grey_dilation`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_dilation.html#scipy.ndimage.grey_dilation)
- [`grey_erosion`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_erosion.html#scipy.ndimage.grey_erosion)
- [`grey_opening`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_opening.html#scipy.ndimage.grey_opening)
- [`histogram`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.histogram.html#scipy.ndimage.histogram)
- [`iterate_structure`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.iterate_structure.html#scipy.ndimage.iterate_structure)
- [`label`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html#scipy.ndimage.label)
- [`labeled_comprehension`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.labeled_comprehension.html#scipy.ndimage.labeled_comprehension)
- [`laplace`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.laplace.html#scipy.ndimage.laplace)
- [`map_coordinates`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html#scipy.ndimage.map_coordinates)
- [`maximum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.maximum.html#scipy.ndimage.maximum)
- [`maximum_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.maximum_filter.html#scipy.ndimage.maximum_filter)
- [`maximum_filter1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.maximum_filter1d.html#scipy.ndimage.maximum_filter1d)
- [`maximum_position`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.maximum_position.html#scipy.ndimage.maximum_position)
- [`mean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.mean.html#scipy.ndimage.mean)
- [`median`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median.html#scipy.ndimage.median)
- [`median_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html#scipy.ndimage.median_filter)
- [`minimum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.minimum.html#scipy.ndimage.minimum)
- [`minimum_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.minimum_filter.html#scipy.ndimage.minimum_filter)
- [`minimum_filter1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.minimum_filter1d.html#scipy.ndimage.minimum_filter1d)
- [`minimum_position`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.minimum_position.html#scipy.ndimage.minimum_position)
- [`morphological_gradient`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.morphological_gradient.html#scipy.ndimage.morphological_gradient)
- [`morphological_laplace`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.morphological_laplace.html#scipy.ndimage.morphological_laplace)
- [`percentile_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.percentile_filter.html#scipy.ndimage.percentile_filter)
- [`prewitt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.prewitt.html#scipy.ndimage.prewitt)
- [`rank_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rank_filter.html#scipy.ndimage.rank_filter)
- [`rotate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html#scipy.ndimage.rotate)
- [`shift`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html#scipy.ndimage.shift)
- [`sobel`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.sobel.html#scipy.ndimage.sobel)
- [`spline_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.spline_filter.html#scipy.ndimage.spline_filter)
- [`spline_filter1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.spline_filter1d.html#scipy.ndimage.spline_filter1d)
- [`standard_deviation`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.standard_deviation.html#scipy.ndimage.standard_deviation)
- `sum`
- [`sum_labels`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.sum_labels.html#scipy.ndimage.sum_labels)
- [`uniform_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter.html#scipy.ndimage.uniform_filter)
- [`uniform_filter1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter1d.html#scipy.ndimage.uniform_filter1d)
- [`value_indices`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.value_indices.html#scipy.ndimage.value_indices)
- [`variance`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.variance.html#scipy.ndimage.variance)
- [`vectorized_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.vectorized_filter.html#scipy.ndimage.vectorized_filter)
- [`watershed_ift`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.watershed_ift.html#scipy.ndimage.watershed_ift)
- [`white_tophat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.white_tophat.html#scipy.ndimage.white_tophat)
- [`zoom`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html#scipy.ndimage.zoom)

## Additional and compatibility namespaces

Many of these paths are deprecated compatibility aliases. Class and object method lists are omitted in these sections to avoid repeating the public API inventory.

### scipy.ndimage.filters

Function entries: **24**. Callable objects: **0**. Classes: **0**.

#### Functions

- `convolve`
- `convolve1d`
- `correlate`
- `correlate1d`
- `gaussian_filter`
- `gaussian_filter1d`
- `gaussian_gradient_magnitude`
- `gaussian_laplace`
- `generic_filter`
- `generic_filter1d`
- `generic_gradient_magnitude`
- `generic_laplace`
- `laplace`
- `maximum_filter`
- `maximum_filter1d`
- `median_filter`
- `minimum_filter`
- `minimum_filter1d`
- `percentile_filter`
- `prewitt`
- `rank_filter`
- `sobel`
- `uniform_filter`
- `uniform_filter1d`

### scipy.ndimage.fourier

Function entries: **4**. Callable objects: **0**. Classes: **0**.

#### Functions

- `fourier_ellipsoid`
- `fourier_gaussian`
- `fourier_shift`
- `fourier_uniform`

### scipy.ndimage.interpolation

Function entries: **8**. Callable objects: **0**. Classes: **0**.

#### Functions

- `affine_transform`
- `geometric_transform`
- `map_coordinates`
- `rotate`
- `shift`
- `spline_filter`
- `spline_filter1d`
- `zoom`

### scipy.ndimage.measurements

Function entries: **17**. Callable objects: **0**. Classes: **0**.

#### Functions

- `center_of_mass`
- `extrema`
- `find_objects`
- `histogram`
- `label`
- `labeled_comprehension`
- `maximum`
- `maximum_position`
- `mean`
- `median`
- `minimum`
- `minimum_position`
- `standard_deviation`
- `sum`
- `sum_labels`
- `variance`
- `watershed_ift`

### scipy.ndimage.morphology

Function entries: **20**. Callable objects: **0**. Classes: **0**.

#### Functions

- `binary_closing`
- `binary_dilation`
- `binary_erosion`
- `binary_fill_holes`
- `binary_hit_or_miss`
- `binary_opening`
- `binary_propagation`
- `black_tophat`
- `distance_transform_bf`
- `distance_transform_cdt`
- `distance_transform_edt`
- `generate_binary_structure`
- `grey_closing`
- `grey_dilation`
- `grey_erosion`
- `grey_opening`
- `iterate_structure`
- `morphological_gradient`
- `morphological_laplace`
- `white_tophat`
