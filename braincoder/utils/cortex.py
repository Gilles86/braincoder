"""Cortical surface helpers: mesh loading and geodesic distances.

* ``load_freesurfer_surface`` -- read an ``lh.white``/``rh.white`` style
  surface as ``(vertices, faces)`` numpy arrays via nibabel.
* ``geodesic_distance_matrix`` -- distances on a triangulated mesh between
  a set of source vertices and a set of target vertices, by one of two
  methods:

  - ``method='dijkstra'`` (default, backwards compatible): shortest path
    along mesh edges. Cheap, but paths are constrained to edges, so
    distances are biased high (typically 5-15 % on irregular meshes).
  - ``method='heat'``: the heat method of Crane, Weischedel & Wardetzky
    (2013), implemented here with scipy sparse LU factorisations only.
    Paths may cut across triangles, so the result approximates the true
    geodesic distance (a few percent error at mesh resolution). This is
    the same algorithm as ``cortex.polyutils.Surface.geodesic_distance``
    (pycortex) and ``potpourri3d.MeshHeatMethodDistanceSolver``, without
    either dependency.

* ``submesh`` -- restrict a mesh to a vertex mask (e.g. the FreeSurfer
  ``cortex.label``, or a neighbourhood of an ROI), which removes the medial
  wall from paths and makes the heat method much cheaper.
* ``block_distance_matrix`` -- stack per-hemisphere distance matrices into
  one matrix with ``inf`` between hemispheres (no geodesic path exists).

No pycortex is needed. scipy is the only dependency (nibabel for loading).
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


def submesh(vertices, faces, mask):
    """Restrict a mesh to the vertices in ``mask``.

    Faces with any vertex outside the mask are dropped.

    Returns
    -------
    vertices_sub : array (k, 3)
    faces_sub : int array (m, 3)
        Re-indexed into ``vertices_sub``.
    index : int array (k,)
        ``index[i]`` is the original vertex index of submesh vertex ``i``.
    """
    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    mask = np.asarray(mask)
    if mask.dtype != bool:
        m = np.zeros(len(vertices), bool)
        m[mask] = True
        mask = m
    index = np.flatnonzero(mask)
    lookup = -np.ones(len(vertices), dtype=np.int64)
    lookup[index] = np.arange(len(index))
    keep = mask[faces].all(axis=1)
    return vertices[index], lookup[faces[keep]], index


def block_distance_matrix(blocks, dtype=np.float32):
    """Combine per-hemisphere distance matrices into one block matrix.

    Off-diagonal blocks (between hemispheres) are ``np.inf``: there is no
    path on the cortical surface between hemispheres, so a Gaussian over
    distance gives them exactly zero weight.

    Parameters
    ----------
    blocks : list of arrays (n_i, m_i)

    Returns
    -------
    D : array (sum n_i, sum m_i)
    """
    blocks = [np.asarray(b) for b in blocks]
    n = sum(b.shape[0] for b in blocks)
    m = sum(b.shape[1] for b in blocks)
    D = np.full((n, m), np.inf, dtype=dtype)
    i = j = 0
    for b in blocks:
        D[i:i + b.shape[0], j:j + b.shape[1]] = b
        i += b.shape[0]
        j += b.shape[1]
    return D


class HeatMethodSolver:
    """Geodesic distances on a triangle mesh via the heat method.

    Crane, Weischedel & Wardetzky (2013), *Geodesics in heat*, ACM TOG.
    Three steps per source vertex:

    1. integrate the heat flow ``(M + t L) u = delta_source`` for a short
       time ``t`` (backward Euler);
    2. normalise the gradient, ``X = -grad u / |grad u|``;
    3. solve the Poisson equation ``L phi = -div X``.

    ``L`` is the (positive semi-definite) cotangent Laplacian and ``M`` the
    lumped (barycentric) mass matrix. Both systems are factorised once with
    ``scipy.sparse.linalg.splu``, so many sources are cheap; they are solved
    in batches (one right-hand side column per source).

    Parameters
    ----------
    vertices : array (n_v, 3)
    faces : int array (n_f, 3)
    t_factor : float
        Heat time as a multiple of the squared mean edge length (Crane et
        al.'s default is 1). Larger values smooth the distance near the
        source; smaller values are more accurate on a good mesh but less
        robust on a poor one.
    """

    def __init__(self, vertices, faces, t_factor=1.0):
        from scipy import sparse
        from scipy.sparse.linalg import splu

        V = np.asarray(vertices, dtype=np.float64)
        F = np.asarray(faces, dtype=np.int64)
        self.n_vertices = n_v = len(V)
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]

        # edges opposite each corner, counter-clockwise
        e0, e1, e2 = v2 - v1, v0 - v2, v1 - v0
        N = np.cross(e2, -e1)                       # (v1 - v0) x (v2 - v0)
        dblA = np.linalg.norm(N, axis=1)
        ok = dblA > 1e-12
        if not ok.all():                            # drop degenerate faces
            F, v0, v1, v2, e0, e1, e2, N, dblA = (
                a[ok] for a in (F, v0, v1, v2, e0, e1, e2, N, dblA))
        unit_n = N / dblA[:, None]

        def cot(a, b):
            return (a * b).sum(1) / np.linalg.norm(np.cross(a, b), axis=1)

        cot0 = cot(v1 - v0, v2 - v0)                 # angle at corner 0
        cot1 = cot(v2 - v1, v0 - v1)
        cot2 = cot(v0 - v2, v1 - v2)

        # cotangent Laplacian (positive semi-definite)
        i = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
        j = np.concatenate([F[:, 2], F[:, 0], F[:, 1]])
        w = 0.5 * np.concatenate([cot0, cot1, cot2])
        W = sparse.coo_matrix((w, (i, j)), shape=(n_v, n_v)).tocsr()
        W = W + W.T
        L = sparse.diags(np.asarray(W.sum(1)).ravel()) - W

        area = 0.5 * dblA
        mass = np.zeros(n_v)
        for k in range(3):
            np.add.at(mass, F[:, k], area / 3.0)
        mass[mass == 0] = mass[mass > 0].min() * 1e-3   # isolated vertices
        M = sparse.diags(mass)

        edge_len = np.concatenate([np.linalg.norm(e, axis=1) for e in (e0, e1, e2)])
        self.mean_edge_length = float(edge_len.mean())
        self.t = t_factor * self.mean_edge_length ** 2

        self._heat = splu((M + self.t * L).tocsc())
        # L is singular (constants); a tiny mass shift makes it invertible
        # without changing distances beyond floating-point noise.
        self._poisson = splu((L + 1e-8 * self.t * M).tocsc())

        # per-face gradient operators: grad u = sum_k u_k * G_k
        self._F = F
        self._G = [np.cross(unit_n, e) / dblA[:, None] for e in (e0, e1, e2)]
        # per-face divergence weights: div_k += a_k . X
        self._A = [0.5 * (cot2[:, None] * (v1 - v0) + cot1[:, None] * (v2 - v0)),
                   0.5 * (cot0[:, None] * (v2 - v1) + cot2[:, None] * (v0 - v1)),
                   0.5 * (cot1[:, None] * (v0 - v2) + cot0[:, None] * (v1 - v2))]
        n_f = len(F)
        self._S = [sparse.csr_matrix((np.ones(n_f), (F[:, k], np.arange(n_f))),
                                     shape=(n_v, n_f)) for k in range(3)]

    def distances(self, sources):
        """Distances from each vertex in ``sources`` to every vertex.

        Returns an array of shape ``(len(sources), n_vertices)``.
        """
        sources = np.atleast_1d(np.asarray(sources, dtype=np.int64))
        k = len(sources)
        delta = np.zeros((self.n_vertices, k))
        delta[sources, np.arange(k)] = 1.0
        u = self._heat.solve(delta)                                  # (n_v, k)
        F = self._F
        grad = sum(u[F[:, c]][:, None, :] * self._G[c][:, :, None]
                   for c in range(3))                                # (n_f, 3, k)
        norm = np.linalg.norm(grad, axis=1, keepdims=True)
        X = -grad / np.maximum(norm, 1e-300)
        div = sum(self._S[c] @ np.einsum('fd,fdk->fk', self._A[c], X)
                  for c in range(3))                                 # (n_v, k)
        phi = self._poisson.solve(-div)
        phi = phi - phi[sources, np.arange(k)][None, :]
        return np.maximum(phi, 0.0).T


def geodesic_distance_matrix(vertices, faces, source_indices=None,
                             target_indices=None, method='dijkstra',
                             max_distance=None, vertex_mask=None,
                             heat_t_factor=1.0, chunk_size=None,
                             progressbar=True, dtype=np.float32):
    """Geodesic distance matrix between source and target vertices.

    Parameters
    ----------
    vertices : array (n_v, 3)
    faces : int array (n_f, 3)
    source_indices : array of int, optional
        Rows of the output (indices into ``vertices``). ``None`` = all
        vertices (expensive).
    target_indices : array of int, optional
        Columns of the output. ``None`` = the same as ``source_indices``;
        the result is then square and symmetrised.
    method : {'dijkstra', 'heat'}
        ``'dijkstra'``: shortest path along mesh edges (scipy only; biased
        high because paths follow edges). ``'heat'``: heat-method geodesic
        distance (Crane et al. 2013; scipy only), which is what you want
        when the distances set a model's spatial scale, e.g. connective
        fields. See :class:`HeatMethodSolver`.
    max_distance : float, optional
        Distances above this are returned as ``np.inf``. For Dijkstra this
        also bounds the search, which is much faster.
    vertex_mask : bool array (n_v,), optional
        Compute on the submesh of these vertices only (e.g. the FreeSurfer
        cortex label, so paths cannot run across the medial wall). Sources
        and targets must lie inside the mask.
    heat_t_factor : float
        Heat time in units of squared mean edge length (``method='heat'``).
    chunk_size : int, optional
        Sources per batch (default 64 for Dijkstra, 128 for heat).
    progressbar : bool
    dtype : numpy dtype

    Returns
    -------
    D : array (n_sources, n_targets)
        Distances in the units of ``vertices`` (mm for FreeSurfer
        surfaces). Pairs with no path (another connected component) are
        ``np.inf``.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    n_v_full = len(vertices)

    if source_indices is None:
        source_indices = np.arange(n_v_full)
    source_indices = np.asarray(source_indices, dtype=np.int64).ravel()
    square = target_indices is None
    target_indices = source_indices if square else \
        np.asarray(target_indices, dtype=np.int64).ravel()

    if vertex_mask is not None:
        vertices, faces, index = submesh(vertices, faces, vertex_mask)
        lookup = -np.ones(n_v_full, dtype=np.int64)
        lookup[index] = np.arange(len(index))
        source_indices, target_indices = lookup[source_indices], lookup[target_indices]
        if (source_indices < 0).any() or (target_indices < 0).any():
            raise ValueError('all source and target vertices must lie inside vertex_mask')

    n = len(source_indices)
    D = np.empty((n, len(target_indices)), dtype=dtype)

    if method == 'dijkstra':
        from scipy.sparse.csgraph import dijkstra
        graph = _build_edge_graph(vertices, faces)
        chunk_size = chunk_size or 64
        solve = lambda srcs: dijkstra(graph, indices=srcs, directed=False,
                                      limit=np.inf if max_distance is None else max_distance)
    elif method == 'heat':
        solver = HeatMethodSolver(vertices, faces, t_factor=heat_t_factor)
        chunk_size = chunk_size or 128
        solve = solver.distances
    else:
        raise ValueError(f"method must be 'dijkstra' or 'heat', got {method!r}")

    chunks = range(0, n, chunk_size)
    if progressbar:
        from tqdm.auto import tqdm
        chunks = tqdm(chunks, desc=f'Geodesic distances ({method})')

    for start in chunks:
        end = min(start + chunk_size, n)
        rows = solve(source_indices[start:end])
        D[start:end, :] = rows[:, target_indices].astype(dtype)

    if method == 'heat':
        # the heat method is not exactly zero at a vertex's own position in
        # other rows, nor exactly symmetric; disconnected parts come out finite
        # but meaningless, so mark them via mesh connectivity
        from scipy.sparse.csgraph import connected_components
        _, comp = connected_components(_build_edge_graph(vertices, faces), directed=False)
        D[comp[source_indices][:, None] != comp[target_indices][None, :]] = np.inf

    if square:
        # Distances are symmetric in theory; the heat method (and float
        # round-trips) leave small asymmetries. Symmetrise defensively.
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)

    if max_distance is not None:
        D[D > max_distance] = np.inf
    return D
