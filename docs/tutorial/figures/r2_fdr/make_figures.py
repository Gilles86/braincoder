"""Generate the figures used in lesson8_r2_fdr.rst.

Run from the repo root::

    python docs/tutorial/figures/r2_fdr/make_figures.py

Writes:
    docs/tutorial/figures/r2_fdr/mixture_diagnostic.png
    docs/tutorial/figures/r2_fdr/posterior_vs_r2.png
    docs/tutorial/figures/r2_fdr/threshold_vs_alpha.png
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from braincoder.utils.stats import (
    fit_r2_mixture, r2_fdr_threshold, posterior_p_signal, plot_r2_mixture,
    _logit, _inv_logit,
)

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
    "figure.titlesize": 14,
})

# Simulate a realistic per-voxel R² distribution: 90% noise + 10% signal.
rng = np.random.default_rng(42)
N_NOISE = 45000
N_SIGNAL = 5000
z_noise = rng.normal(loc=-5.8, scale=0.7, size=N_NOISE)
z_signal = rng.normal(loc=-3.0, scale=1.6, size=N_SIGNAL)
z = np.concatenate([z_noise, z_signal])
r2 = _inv_logit(z)
r2 = r2[(r2 > 0) & (r2 < 0.99)]
print(f"Simulated n_voxels={len(r2)}")

# Fit
fit = fit_r2_mixture(r2)
thr_05 = r2_fdr_threshold(fit, alpha=0.05)
thr_01 = r2_fdr_threshold(fit, alpha=0.01)
print(f"α=0.05 → R²≥{thr_05:.4f}")
print(f"α=0.01 → R²≥{thr_01:.4f}")

OUT = Path(__file__).parent

# ---------- Figure 1: canonical diagnostic plot ----------
fig, ax = plt.subplots(figsize=(7.5, 4.6))
plot_r2_mixture(fit, r2=r2, alpha=0.05, ax=ax,
                 title='Per-voxel R² with fitted 2-component mixture')
ax.set_ylim(bottom=1e-3)
fig.tight_layout()
fig.savefig(OUT / 'mixture_diagnostic.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print('wrote mixture_diagnostic.png')

# ---------- Figure 2: posterior vs R² ----------
p_sig = posterior_p_signal(r2, fit)
order = np.argsort(r2)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(r2[order], p_sig[order], color='#1f77b4', lw=1.5)
for thr, alpha, color, ls in [(thr_05, 0.05, 'k', ':'),
                                (thr_01, 0.01, '#555', '-.')]:
    ax.axvline(thr, color=color, lw=1.2, ls=ls,
               label=f'α={alpha} thr (R²={thr:.3f})')
ax.axhline(0.95, color='#d62728', lw=1, ls=(0, (4, 2)),
           label='p_signal=0.95 cutoff')
ax.set_xscale('log')
r2_ticks = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
ax.set_xticks(r2_ticks); ax.set_xticklabels([f'{t:g}' for t in r2_ticks])
ax.set_xlabel('R²  (log axis)')
ax.set_ylabel('Posterior P(signal | R²)')
ax.set_title('Per-voxel posterior vs. R²')
ax.legend(loc='center left', fontsize=9)
ax.set_ylim(-0.02, 1.02)
fig.tight_layout()
fig.savefig(OUT / 'posterior_vs_r2.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print('wrote posterior_vs_r2.png')

# ---------- Figure 3: threshold vs α ----------
alphas = np.geomspace(1e-4, 0.5, 60)
thrs = np.array([r2_fdr_threshold(fit, alpha=a) for a in alphas])
n_above = np.array([(r2 >= t).sum() for t in thrs])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
axes[0].plot(alphas, thrs, color='#1f77b4', lw=2)
axes[0].axhline(thr_05, color='k', lw=0.8, ls=':')
axes[0].axhline(thr_01, color='#555', lw=0.8, ls='-.')
axes[0].set_xscale('log'); axes[0].set_xlabel('α (tail-FDR)')
axes[0].set_ylabel('R² threshold'); axes[0].set_title('Threshold vs. α')
axes[1].plot(alphas, 100 * n_above / len(r2), color='#d62728', lw=2)
axes[1].set_xscale('log'); axes[1].set_xlabel('α (tail-FDR)')
axes[1].set_ylabel('% of voxels above threshold')
axes[1].set_title('Surviving voxels vs. α')
fig.suptitle('Stricter α → higher R² threshold → fewer voxels kept', weight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / 'threshold_vs_alpha.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print('wrote threshold_vs_alpha.png')

# ---------- Figure 4: real data (Szinte 2024 V1) ----------
# Apply the same machinery to the pre-fit V1 R²s shipped with braincoder.
from braincoder.utils.data import load_szinte2024

d = load_szinte2024()
r2_szinte = d['r2'].values
r2_szinte = r2_szinte[np.isfinite(r2_szinte) & (r2_szinte > 0) & (r2_szinte < 0.99)]
print(f"Szinte n_voxels={len(r2_szinte)}  median R²={np.median(r2_szinte):.3f}")
fit_sz = fit_r2_mixture(r2_szinte)
thr_sz_05 = r2_fdr_threshold(fit_sz, alpha=0.05)

fig, ax = plt.subplots(figsize=(7.5, 4.6))
plot_r2_mixture(fit_sz, r2=r2_szinte, alpha=0.05, ax=ax,
                 title=f'Szinte 2024 V1 R² (n={len(r2_szinte)}) — '
                       f'2-component logit-Gaussian fit')
ax.set_ylim(bottom=1e-3)
fig.tight_layout()
fig.savefig(OUT / 'szinte_v1_mixture.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print('wrote szinte_v1_mixture.png')
