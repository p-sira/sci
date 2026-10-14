# scipy.cluster

[Back to SciPy function index](../SCIPY_FUNCTIONS.md)

## Namespaces

- [`scipy.cluster`](#scipy-cluster)
- [`scipy.cluster.hierarchy`](#scipy-cluster-hierarchy)
- [`scipy.cluster.vq`](#scipy-cluster-vq)

## Public API namespaces

### scipy.cluster

[Module reference](https://docs.scipy.org/doc/scipy/reference/cluster.html#module-scipy.cluster)

Function entries: **0**. Callable objects: **0**. Classes: **0**.

This namespace contains submodules or has no exported callable entries in this runtime.

### scipy.cluster.hierarchy

[Module reference](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html#module-scipy.cluster.hierarchy)

Function entries: **30**. Callable objects: **0**. Classes: **3**.

#### Functions

- [`average`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.average.html#scipy.cluster.hierarchy.average)
- [`centroid`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.centroid.html#scipy.cluster.hierarchy.centroid)
- [`complete`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.complete.html#scipy.cluster.hierarchy.complete)
- [`cophenet`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.cophenet.html#scipy.cluster.hierarchy.cophenet)
- [`correspond`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.correspond.html#scipy.cluster.hierarchy.correspond)
- [`cut_tree`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.cut_tree.html#scipy.cluster.hierarchy.cut_tree)
- [`dendrogram`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html#scipy.cluster.hierarchy.dendrogram)
- [`fcluster`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster)
- [`fclusterdata`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fclusterdata.html#scipy.cluster.hierarchy.fclusterdata)
- [`from_mlab_linkage`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.from_mlab_linkage.html#scipy.cluster.hierarchy.from_mlab_linkage)
- [`inconsistent`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.inconsistent.html#scipy.cluster.hierarchy.inconsistent)
- [`is_isomorphic`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.is_isomorphic.html#scipy.cluster.hierarchy.is_isomorphic)
- [`is_monotonic`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.is_monotonic.html#scipy.cluster.hierarchy.is_monotonic)
- [`is_valid_im`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.is_valid_im.html#scipy.cluster.hierarchy.is_valid_im)
- [`is_valid_linkage`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.is_valid_linkage.html#scipy.cluster.hierarchy.is_valid_linkage)
- [`leaders`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.leaders.html#scipy.cluster.hierarchy.leaders)
- [`leaves_list`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.leaves_list.html#scipy.cluster.hierarchy.leaves_list)
- [`linkage`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage)
- [`maxRstat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.maxRstat.html#scipy.cluster.hierarchy.maxrstat)
- [`maxdists`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.maxdists.html#scipy.cluster.hierarchy.maxdists)
- [`maxinconsts`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.maxinconsts.html#scipy.cluster.hierarchy.maxinconsts)
- [`median`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.median.html#scipy.cluster.hierarchy.median)
- [`num_obs_linkage`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.num_obs_linkage.html#scipy.cluster.hierarchy.num_obs_linkage)
- [`optimal_leaf_ordering`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.optimal_leaf_ordering.html#scipy.cluster.hierarchy.optimal_leaf_ordering)
- [`set_link_color_palette`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.set_link_color_palette.html#scipy.cluster.hierarchy.set_link_color_palette)
- [`single`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.single.html#scipy.cluster.hierarchy.single)
- [`to_mlab_linkage`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.to_mlab_linkage.html#scipy.cluster.hierarchy.to_mlab_linkage)
- [`to_tree`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.to_tree.html#scipy.cluster.hierarchy.to_tree)
- [`ward`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.ward.html#scipy.cluster.hierarchy.ward)
- [`weighted`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.weighted.html#scipy.cluster.hierarchy.weighted)

#### Classes and their public methods

- [`ClusterNode`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.ClusterNode.html#scipy.cluster.hierarchy.clusternode)
  - [`ClusterNode.get_count`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.ClusterNode.get_count.html#scipy.cluster.hierarchy.clusternode.get_count)
  - [`ClusterNode.get_id`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.ClusterNode.get_id.html#scipy.cluster.hierarchy.clusternode.get_id)
  - [`ClusterNode.get_left`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.ClusterNode.get_left.html#scipy.cluster.hierarchy.clusternode.get_left)
  - [`ClusterNode.get_right`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.ClusterNode.get_right.html#scipy.cluster.hierarchy.clusternode.get_right)
  - [`ClusterNode.is_leaf`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.ClusterNode.is_leaf.html#scipy.cluster.hierarchy.clusternode.is_leaf)
  - [`ClusterNode.pre_order`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.ClusterNode.pre_order.html#scipy.cluster.hierarchy.clusternode.pre_order)
- [`ClusterWarning`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.ClusterWarning.html#scipy.cluster.hierarchy.clusterwarning)
  - [`ClusterWarning.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.ClusterWarning.html#scipy.cluster.hierarchy.clusterwarning.add_note)
  - [`ClusterWarning.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.ClusterWarning.html#scipy.cluster.hierarchy.clusterwarning.with_traceback)
- [`DisjointSet`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.DisjointSet.html#scipy.cluster.hierarchy.disjointset)
  - [`DisjointSet.add`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.DisjointSet.add.html#scipy.cluster.hierarchy.disjointset.add)
  - [`DisjointSet.connected`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.DisjointSet.connected.html#scipy.cluster.hierarchy.disjointset.connected)
  - [`DisjointSet.merge`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.DisjointSet.merge.html#scipy.cluster.hierarchy.disjointset.merge)
  - [`DisjointSet.subset`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.DisjointSet.subset.html#scipy.cluster.hierarchy.disjointset.subset)
  - [`DisjointSet.subset_size`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.DisjointSet.subset_size.html#scipy.cluster.hierarchy.disjointset.subset_size)
  - [`DisjointSet.subsets`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.DisjointSet.subsets.html#scipy.cluster.hierarchy.disjointset.subsets)

### scipy.cluster.vq

[Module reference](https://docs.scipy.org/doc/scipy/reference/cluster.vq.html#module-scipy.cluster.vq)

Function entries: **5**. Callable objects: **0**. Classes: **1**.

#### Functions

- [`kmeans`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans.html#scipy.cluster.vq.kmeans)
- [`kmeans2`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans2.html#scipy.cluster.vq.kmeans2)
- `py_vq`
- [`vq`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.vq.html#scipy.cluster.vq.vq)
- [`whiten`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.whiten.html#scipy.cluster.vq.whiten)

#### Classes and their public methods

- [`ClusterError`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.ClusterError.html#scipy.cluster.vq.clustererror)
  - [`ClusterError.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.ClusterError.html#scipy.cluster.vq.clustererror.add_note)
  - [`ClusterError.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.ClusterError.html#scipy.cluster.vq.clustererror.with_traceback)
