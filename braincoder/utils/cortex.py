"""Cortical surface helpers for the GP-prior workflow.

Two thin wrappers:

  * ``load_freesurfer_surface`` — read an ``lh.white``/``rh.white`` style
    surface as ``(vertices, faces)`` numpy arrays via nibabel.
  * ``geodesic_distance_matrix`` — compute a pairwise distance matrix
    on a triangular mesh by running Dijkstra on the mesh graph with
    edge weights set to Euclidean edge lengths.

Neither pycortex nor a full Laplace-Beltrami implementation is needed.
The mesh-edge Dijkstra approximation biases distances slightly high
(paths are constrained to follow edges instead of cutting across
triangles, typically a 10-20% overestimate). For our use — an RBF
kernel ``exp(-d^2 / 2l^2)`` whose lengthscale is fit empirically from
the data — this bias gets absorbed into the fitted lengthscale and is
invisible to the prior.
"""

import os

import numpy as np


def load_freesurfer_surface(subject, hemi, surface='white', subjects_dir=None):
    """Load a FreeSurfer surface as ``(vertices, faces)`` numpy arrays.

    Parameters
    ----------
    subject : str
        FreeSurfer subject ID (the directory name under ``$SUBJECTS_DIR``).
    hemi : {'lh', 'rh'}
    surface : str
        Surface name. ``'white'``, ``'pial'``, ``'midgray'`` etc.
    subjects_dir : str, optional
        Defaults to ``$SUBJECTS_DIR``.
    """
    import nibabel as nib  # optional dependency

    if hemi not in ('lh', 'rh'):
        raise ValueError(f"hemi must be 'lh' or 'rh', got {hemi!r}")
    if subjects_dir is None:
        subjects_dir = os.environ.get('SUBJECTS_DIR')
        if subjects_dir is None:
            raise ValueError(
                "subjects_dir not given and SUBJECTS_DIR not set")

    path = os.path.join(subjects_dir, subject, 'surf', f'{hemi}.{surface}')
    vertices, faces = nib.freesurfer.read_geometry(path)
    return (np.asarray(vertices, dtype=np.float32),
            np.asarray(faces, dtype=np.int64))


def _build_edge_graph(vertices, faces):
    """Build a sparse symmetric weighted-adjacency graph from a mesh.

    Each undirected mesh edge becomes one entry per direction with
    weight = Euclidean length. Interior edges are shared by two
    triangles, so we deduplicate by sorted (i, j) before building the
    sparse matrix — otherwise coo→csr would sum duplicate entries and
    double the edge weights.
    """
    from scipy.sparse import csr_matrix

    n_v = len(vertices)
    e = np.concatenate([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [0, 2]],
    ], axis=0)
    e = np.sort(e, axis=1)
    e = np.unique(e, axis=0)
    diff = vertices[e[:, 0]] - vertices[e[:, 1]]
    w = np.sqrt((diff ** 2).sum(axis=1)).astype(np.float64)
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    data = np.concatenate([w, w])
    return csr_matrix((data, (rows, cols)), shape=(n_v, n_v))


def geodesic_distance_matrix(vertices, faces, source_indices=None,
                             progressbar=True, dtype=np.float32):
    """Pairwise mesh-graph (Dijkstra) distance matrix.

    Approximates geodesic distance by the shortest path along mesh
    edges. Lightweight (scipy only) and sufficient for an RBF-kernel
    GP prior where the lengthscale is fit empirically.

    Parameters
    ----------
    vertices : array (n_v, 3)
    faces : int array (n_f, 3)
    source_indices : array of int, optional
        Indices into ``vertices`` defining the ROI. If given, returns
        ``(k, k)`` pairwise distances among these vertices. If ``None``,
        the full ``(n_v, n_v)`` matrix is returned (expensive).
    progressbar : bool
        Show a tqdm bar while iterating Dijkstra batches.
    dtype : numpy dtype
        Output dtype. ``float32`` saves half the memory and is plenty
        accurate for distances in mm.

    Returns
    -------
    D : array (k, k)
        Symmetric pairwise distances in the same units as ``vertices``
        (millimeters for FreeSurfer surfaces), with zero diagonal.
    """
    from scipy.sparse.csgraph import dijkstra

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    if source_indices is None:
        source_indices = np.arange(len(vertices))
    source_indices = np.asarray(source_indices, dtype=int).ravel()
    n = len(source_indices)

    graph = _build_edge_graph(vertices, faces)

    # Multi-source Dijkstra is fast; chunk to keep peak memory bounded
    # for big ROIs. ``indices`` returns one row of distances per source.
    chunk_size = 64
    chunks = range(0, n, chunk_size)
    if progressbar:
        from tqdm.auto import tqdm
        chunks = tqdm(chunks, desc='Mesh Dijkstra')

    D = np.zeros((n, n), dtype=dtype)
    for start in chunks:
        end = min(start + chunk_size, n)
        srcs = source_indices[start:end]
        # rows: (len(srcs), n_v) — distance from each src to every vertex
        rows = dijkstra(graph, indices=srcs, directed=False)
        D[start:end, :] = rows[:, source_indices].astype(dtype)

    # Dijkstra on a symmetric graph is already symmetric, but float
    # round-trip can leave tiny asymmetries. Symmetrize defensively.
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D
