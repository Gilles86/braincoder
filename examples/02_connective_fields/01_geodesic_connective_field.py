"""
==========================================================
Connective fields with geodesic distance on a folded cortex
==========================================================

A connective field (CF; Haak et al., 2013) describes a target vertex's
activity as a Gaussian-weighted sum of the activity in a *source* region,
for example V1:

.. math::

    \\hat y(t) = \\beta_0 + \\beta_1 \\sum_s
    \\exp\\left(-\\frac{d(c, s)^2}{2\\sigma^2}\\right) x_s(t)

The fitted centre :math:`c` is a source vertex, so any property of the
source vertices (e.g. visual-field position from a retinotopic atlas) can
be read off for the target.

The distance :math:`d` must be measured **along the cortical sheet**. Two
banks of a sulcus can be 2 mm apart in 3-D and 30 mm apart on the surface;
a Euclidean Gaussian pools across the fold. Here we build a toy "sulcus",
show the problem, and fit CFs with
:class:`braincoder.models.GaussianConnectiveField` and
:class:`braincoder.optimize.ConnectiveFieldFitter`.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from braincoder.models import GaussianConnectiveField
from braincoder.optimize import ConnectiveFieldFitter
from braincoder.utils.cortex import geodesic_distance_matrix

# %%
# A folded source region
# ----------------------
# A strip of cortex (1 mm grid) folded into a sulcus: bank A at z = 0, a
# half-turn of 1 mm radius, and bank B running back 2 mm above bank A.
nu, nv, bank, radius = 34, 8, 15.0, 1.0
u, v = [a.ravel() for a in np.meshgrid(np.arange(nu, dtype=float),
                                       np.arange(nv, dtype=float), indexing='ij')]
turn = np.pi * radius
x = np.where(u <= bank, u, np.where(u <= bank + turn,
                                    bank + radius * np.sin((u - bank) / radius),
                                    bank - (u - bank - turn)))
z = np.where(u <= bank, 0., np.where(u <= bank + turn,
                                     radius * (1 - np.cos((u - bank) / radius)), 2 * radius))
vertices = np.c_[x, v, z]
faces = np.array([f for i in range(nu - 1) for j in range(nv - 1)
                  for f in ([[i * nv + j, (i + 1) * nv + j, (i + 1) * nv + j + 1],
                             [i * nv + j, (i + 1) * nv + j + 1, i * nv + j + 1]])])

# %%
# Geodesic distances use the heat method (Crane et al., 2013), with scipy
# only. For FreeSurfer surfaces, pass the midthickness mesh, restrict to the
# cortex label with ``vertex_mask`` and use one matrix per hemisphere
# (:func:`braincoder.utils.cortex.block_distance_matrix` puts ``inf``
# between them).
D_geo = geodesic_distance_matrix(vertices, faces, method='heat', progressbar=False)
D_euc = np.linalg.norm(vertices[:, None] - vertices[None], axis=2)

centre = int(np.flatnonzero((u == 12) & (v == 4))[0])
fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
for ax, D, name in zip(axes, (D_geo, D_euc), ('Geodesic', 'Euclidean')):
    w = np.exp(-D[centre] ** 2 / (2 * 2.0 ** 2))
    ax.scatter(u, v, c=w, cmap='magma', s=40, vmin=0, vmax=1)
    ax.axvspan(bank, bank + turn, color='0.85', zorder=-1)
    ax.set(title=f'{name} Gaussian, sigma = 2 mm', xlabel='Position along the surface (mm)')
axes[0].set_ylabel('Width (mm)')
fig.tight_layout()

# %%
# Sharing weight across the fold (right) means a CF centred on bank A also
# pools bank B, which in visual cortex is often the other side of a
# retinotopic map.

# %%
# Simulate sources and targets
# ----------------------------
# Source signals are smooth along the surface; each target is a geodesic CF
# of the sources plus noise.
rng = np.random.default_rng(0)
n_samples = 500
K = np.exp(-D_geo ** 2 / (2 * 1.5 ** 2))
sources = rng.normal(size=(n_samples, len(vertices))) @ K.T
sources = (sources - sources.mean(0)) / sources.std(0)

centres = rng.choice(np.flatnonzero((v >= 1) & (v <= 6)), 60, replace=False)
true = pd.DataFrame({'center_index': centres, 'sigma': rng.choice([1., 2., 4.], 60),
                     'baseline': 0., 'amplitude': 1.}).astype(np.float32)
model = GaussianConnectiveField(D_geo, normalize='sum', truncate=3.0)
data = model.predict(sources, true)
data += 0.5 * data.std().values * rng.normal(size=data.shape)

# %%
# Fit: grid search over centres x sizes, then refine sigma
# --------------------------------------------------------
# The grid search maximises the correlation between each target and every
# candidate's pooled source signal. With a free baseline and a positive
# amplitude this is the least-squares solution too; amplitude and baseline
# follow in closed form. The discrete centre is then held fixed while sigma,
# baseline and amplitude are refined by gradient descent.
results = {}
for name, D in (('Geodesic', D_geo), ('Euclidean', D_euc)):
    m = GaussianConnectiveField(D, normalize='sum', truncate=3.0)
    fitter = ConnectiveFieldFitter(m, data, sources)
    grid = fitter.fit_grid(sigmas=[1., 2., 4.], progressbar=False)
    results[name] = dict(pars=grid, r2=fitter.grid_rsq_,
                         cv=fitter.fit_cv(np.repeat(np.arange(5), 100), [1., 2., 4.]))
    if name == 'Geodesic':
        refined = fitter.refine(grid, max_n_iterations=500, progressbar=False)

# %%
# Map each fitted centre to a property of its source vertex. Here: its
# position along the strip, and whether it lies on bank A or bank B.
props = pd.DataFrame({'position': u, 'bank': np.where(u < bank + turn / 2, 'A', 'B')})
for name, res in results.items():
    est = model.source_properties(res['pars'], props)
    ok = (est['bank'].values == props['bank'].values[centres]).mean()
    err = np.abs(est['position'].values - u[centres])
    print(f'{name:9s}  median centre error {np.median(err):.1f} mm, '
          f'correct bank {ok:.2f}, mean cvR2 {res["cv"]["cv_r2"].mean():.3f}')

fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(true['sigma'] + rng.normal(0, .05, 60), refined['sigma'], s=12)
ax.plot([0, 5], [0, 5], color='k', lw=.8)
ax.set(xlabel='True sigma (mm)', ylabel='Refined sigma (mm)', title='Gradient-refined CF size')
fig.tight_layout()
