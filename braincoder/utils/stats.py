"""Statistics utilities for PRF / encoding-model R² maps.

Two related-but-distinct 2-component R²-mixture classifiers live in this
module. **Default to** :func:`fit_r2_mixture` **(logit-Gaussian)** unless
you have a specific reason to prefer F+Beta.

- :func:`fit_r2_mixture` / :func:`r2_fdr_threshold` / :func:`plot_r2_mixture`
  fit a 2-component Gaussian mixture on ``logit(R²)``. Both components are
  free (means + variances + weights). The logit transform stretches
  R²∈[0,1] to ℝ, so a wide signal tail and a sharp noise peak are both
  representable without contortions. Empirically (retsupp 7T PRF, 30 subj,
  2026-05) this gives sensible bimodal fits at every retinotopic ROI *and*
  whole-brain, with BIC strongly preferring K=2 (ΔBIC ≳ 1500 vs K=3 on
  N≈300k voxels). Use for visualization, whole-brain thresholding, and
  per-ROI FDR.

- :func:`fit_r2_f_beta_mixture` / :func:`r2_fdr_threshold_f_beta` /
  :func:`plot_r2_f_beta_mixture` fit a model-anchored mixture on R² directly:
  **noise** = F(d1, d2) ≡ Beta(d1/2, d2/2) with d1 *fixed at the number of
  free per-voxel PRF parameters k*; d2 fit by EM. **signal** = Beta(α_s, β_s)
  with both shapes free. The fixed d1 pins the upper-tail shape of the null
  using model complexity, so signal/noise separation can be sharper in
  low-SNR ROIs where the logit-Gaussian "signal" component blends into
  the noise. Use only if you need a model-anchored null (e.g. publication-
  grade per-voxel FDR in a noisy ROI).

When to use each:
- Whole-brain or GM-wide thresholding for visualization → ``fit_r2_mixture``.
- Per-ROI FDR thresholding for downstream model fits → ``fit_r2_mixture``
  (matches naive 2-Beta within Jaccard ≥0.83 on retinotopic ROIs).
- Per-voxel FDR reporting in a ROI where the logit-Gaussian "signal" is
  poorly separated → ``fit_r2_f_beta_mixture``.

Avoid: an all-free **2-Beta** mixture on R² (i.e. both Betas with α, β free).
The signal Beta degenerates to ≈ uniform when the noise right-tail is heavy,
flagging up to a third of the brain as "signal" at R²≈0.004 (observed on
retsupp whole-brain, 2026-05). Either anchor the noise (F+Beta above) or
work in logit-Gaussian space.

Likelihood note: the logit-Gaussian LL is on ``logit(R²)`` scale; the F-Beta
LL is on ``R²`` scale. To compare them on the same scale, add the Jacobian
correction ``+ sum(log r²ᵢ + log(1−r²ᵢ))`` to the logit LL.
"""
import numpy as np


def _logit(r2):
    return np.log(r2 / (1.0 - r2))


def _inv_logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_r2_mixture(r2, n_init=8, max_iter=500, seed=0):
    """Fit a 2-component Gaussian mixture on ``logit(R²) = log(R²/(1-R²))``.

    The logit pulls apart the low-R² region so the noise and signal
    components don't overlap on a near-singular support, and Gaussians
    (unlike Betas) can't go pathologically U-shaped.

    Args:
        r2: array-like of per-voxel/per-vertex R² values.
        n_init: number of EM restarts (sklearn default = 1; 8 is safer).
        max_iter: max EM iterations per restart.
        seed: random state.

    Returns:
        ``dict`` with keys
        ``{mixture, noise_mu, noise_sigma, noise_weight, noise_mean_r2,
        signal_mu, signal_sigma, signal_weight, signal_mean_r2,
        log_likelihood, n_voxels}``. ``_mu``/``_sigma`` are on the
        logit scale; ``_mean_r2`` is ``inv_logit(_mu)``.

    Raises:
        ValueError: fewer than 50 finite R² values in (0, 0.99).
    """
    from sklearn.mixture import GaussianMixture
    r2 = np.asarray(r2, dtype=float).ravel()
    r2 = r2[np.isfinite(r2) & (r2 > 0) & (r2 < 0.99)]
    if len(r2) < 50:
        raise ValueError(f"need ≥50 finite R² values in (0, 0.99); got {len(r2)}")
    r2 = np.clip(r2, 1e-6, 1 - 1e-6)
    z = _logit(r2).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, n_init=n_init,
                          max_iter=max_iter, random_state=seed,
                          reg_covar=1e-6).fit(z)
    means = gmm.means_.flatten()
    sds = np.sqrt(gmm.covariances_.flatten())
    ws = gmm.weights_
    n_idx, s_idx = int(np.argmin(means)), int(np.argmax(means))
    return {
        'mixture':       'gmm_logit',
        'noise_mu':      float(means[n_idx]),
        'noise_sigma':   float(sds[n_idx]),
        'noise_weight':  float(ws[n_idx]),
        'noise_mean_r2': float(_inv_logit(means[n_idx])),
        'signal_mu':      float(means[s_idx]),
        'signal_sigma':   float(sds[s_idx]),
        'signal_weight':  float(ws[s_idx]),
        'signal_mean_r2': float(_inv_logit(means[s_idx])),
        'log_likelihood': float(gmm.score(z) * len(z)),
        'n_voxels':       int(len(r2)),
    }


def r2_fdr_threshold(r2_or_fit, alpha=0.05, n_grid=4000):
    """R² threshold at which the 2-component mixture's tail-FDR is ≤ α.

        FDR(t) = w_n · P(R² > t | noise) /
                 [w_n · P(R² > t | noise) + w_s · P(R² > t | signal)]

    Accepts either an R² array (fits a fresh mixture via
    :func:`fit_r2_mixture`) or a previously-fitted dict (faster — no
    refit). Returns ``np.inf`` if no threshold in (0, 1) achieves the
    requested α (usually a near-degenerate mixture).
    """
    from scipy.stats import norm
    if not isinstance(r2_or_fit, dict):
        fit = fit_r2_mixture(r2_or_fit)
    else:
        fit = r2_or_fit
    z_grid = np.linspace(fit['noise_mu'] - 5 * fit['noise_sigma'],
                          fit['signal_mu'] + 8 * fit['signal_sigma'],
                          n_grid)
    sf_n = 1.0 - norm.cdf(z_grid, fit['noise_mu'],  fit['noise_sigma'])
    sf_s = 1.0 - norm.cdf(z_grid, fit['signal_mu'], fit['signal_sigma'])
    denom = fit['noise_weight'] * sf_n + fit['signal_weight'] * sf_s
    fdr = np.where(denom > 1e-12,
                   fit['noise_weight'] * sf_n / np.maximum(denom, 1e-12),
                   1.0)
    hits = np.where(fdr <= alpha)[0]
    if len(hits) == 0:
        return float('inf')
    return float(_inv_logit(z_grid[hits[0]]))


def plot_r2_mixture(fit, r2=None, alpha=None, threshold=None, ax=None,
                     title=None):
    """Diagnostic plot for :func:`fit_r2_mixture`.

    Histogram of ``logit(R²)`` with the two Gaussian component PDFs and
    (optionally) the FDR threshold overlaid. X-ticks are labelled on
    the raw R² scale for readability. Pass ``alpha`` to compute the
    threshold from the mixture, or pass ``threshold`` directly.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    z_lo = fit['noise_mu'] - 4 * fit['noise_sigma']
    z_hi = fit['signal_mu'] + 4 * fit['signal_sigma']

    if r2 is not None:
        r2 = np.asarray(r2, dtype=float).ravel()
        r2 = r2[np.isfinite(r2) & (r2 > 0) & (r2 < 0.99)]
        z_data = _logit(np.clip(r2, 1e-6, 1 - 1e-6))
        z_lo = min(z_lo, float(np.percentile(z_data, 0.1)))
        z_hi = max(z_hi, float(np.percentile(z_data, 99.9)))
        ax.hist(z_data, bins=80, density=True, color='0.85',
                edgecolor='0.5', alpha=0.9,
                label=f'data (n={len(r2)})')
    ax_yscale = 'linear'

    z_grid = np.linspace(z_lo, z_hi, 500)
    p_n = (fit['noise_weight']
           * norm.pdf(z_grid, fit['noise_mu'], fit['noise_sigma']))
    p_s = (fit['signal_weight']
           * norm.pdf(z_grid, fit['signal_mu'], fit['signal_sigma']))
    ax.plot(z_grid, p_n, color='#1f77b4', lw=2,
            label=f"Noise (w={fit['noise_weight']:.2f})")
    ax.plot(z_grid, p_s, color='#d62728', lw=2,
            label=f"Signal (w={fit['signal_weight']:.2f})")
    ax.plot(z_grid, p_n + p_s, color='k', lw=1, ls='--', alpha=0.6)

    if threshold is None and alpha is not None:
        threshold = r2_fdr_threshold(fit, alpha=alpha)
    if threshold is not None and np.isfinite(threshold):
        z_thr = _logit(np.clip(threshold, 1e-6, 1 - 1e-6))
        label = f'Threshold R²={threshold:.3f}'
        if alpha is not None:
            label += f' (α={alpha})'
        ax.axvline(z_thr, color='k', ls=':', lw=1.3, label=label)

    r2_ticks_all = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    z_ticks = [_logit(t) for t in r2_ticks_all]
    keep = [(z, t) for z, t in zip(z_ticks, r2_ticks_all) if z_lo <= z <= z_hi]
    if keep:
        ax.set_xticks([z for z, _ in keep])
        ax.set_xticklabels([f'{t:g}' for _, t in keep])
    ax.set_xlabel('R²  (logit-scaled axis)')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right', fontsize=9)
    if title is not None:
        ax.set_title(title)
    return fig


def _weighted_beta_mom(weights, x):
    """Method-of-moments Beta(α, β) fit from weighted samples on (0, 1)."""
    w_sum = weights.sum()
    if w_sum < 1e-9:
        return None
    mu = (weights * x).sum() / w_sum
    var = (weights * (x - mu) ** 2).sum() / w_sum
    if var <= 1e-9 or mu <= 0 or mu >= 1:
        return None
    nu = mu * (1 - mu) / var - 1
    if nu <= 0:
        return None
    return max(0.5, mu * nu), max(0.5, (1 - mu) * nu)


def fit_r2_f_beta_mixture(r2, d1_noise=None, max_iter=400, n_init=12,
                          tol=1e-7, seed=0):
    """2-component mixture on R² ∈ (0, 1) with an F-distributed noise component.

    - **Noise**: ``R² ~ Beta(d1/2, d2/2)``, equivalent under the transformation
      ``F = R²/(1-R²) * d2/d1`` to ``F ~ F-distribution(d1, d2)``. Under the
      classical null with k linear parameters and n trials, d1=k and d2=n-k.
    - **Signal**: ``R² ~ Beta(α_s, β_s)``, both free. Constraint that signal
      mean exceeds noise mean enforced per restart.

    Parameters
    ----------
    d1_noise : float or None
        If given, fix the noise F-distribution's numerator dof (d1 = 2·α_noise)
        to this value — the model-complexity dof. Recommended:
        d1 = number of free per-voxel PRF parameters (e.g. k=5 for a 1D
        log-Gaussian PRF: μ, σ, weight, amplitude, baseline). Only the
        denominator dof (d2 = 2·β_noise) is then fit by EM, which keeps the
        upper-tail shape of the null pinned by model complexity while letting
        d2 < (n - k) absorb fMRI temporal autocorrelation.
        ``None`` (default) leaves both noise dofs free — in which case the
        noise component is just a free Beta and the F-labelling is cosmetic.

    Returns a dict with keys::

        {'mixture': 'f_beta',
         'noise_alpha', 'noise_beta', 'noise_dof1', 'noise_dof2',
         'noise_d1_fixed',  # echoes the input d1_noise
         'noise_mean_r2', 'noise_weight',
         'signal_alpha', 'signal_beta', 'signal_mean_r2', 'signal_weight',
         'log_likelihood', 'n_voxels'}

    Raises ``ValueError`` on fewer than 50 usable voxels or all restarts failing.
    """
    from scipy.stats import beta as beta_dist
    r2 = np.asarray(r2, dtype=np.float64).ravel()
    r2 = r2[np.isfinite(r2) & (r2 > 0) & (r2 < 0.99)]
    if len(r2) < 50:
        raise ValueError(
            f"need >= 50 finite R² values in (0, 0.99); got {len(r2)}")
    r2 = np.clip(r2, 1e-6, 1 - 1e-6)
    rng = np.random.default_rng(seed)
    median_r2 = float(np.median(r2))

    fixed_alpha_n = (d1_noise / 2.0) if d1_noise is not None else None

    def _build_noise(a_free, b_free, mu_lo=None):
        """Honor d1_noise: noise α is fixed if requested; β derived from mean."""
        if fixed_alpha_n is None:
            return a_free, b_free
        a = fixed_alpha_n
        mu = a_free / (a_free + b_free) if mu_lo is None else mu_lo
        mu = float(np.clip(mu, 1e-6, 0.999))
        b = a * (1 - mu) / mu
        return a, b

    # Build a diverse set of (noise / signal) inits anchored on data quantiles
    # so the signal component finds the upper tail even when its weight is tiny.
    inits = []
    for q_lo_hi, w_signal in [
        ((0.50, 0.99), 0.05),
        ((0.50, 0.99), 0.02),
        ((0.70, 0.99), 0.01),
        ((0.30, 0.95), 0.10),
        ((0.50, 0.97), 0.03),
    ]:
        q_lo, q_hi = q_lo_hi
        lo = r2[r2 < np.quantile(r2, q_lo)]
        hi = r2[r2 > np.quantile(r2, q_hi)]
        if len(lo) < 5 or len(hi) < 5:
            continue
        n_init_n = _weighted_beta_mom(np.ones(len(lo)), lo)
        s_init = _weighted_beta_mom(np.ones(len(hi)), hi)
        if n_init_n is None or s_init is None:
            continue
        a_n, b_n = _build_noise(*n_init_n)
        a_s, b_s = s_init
        if a_s / (a_s + b_s) <= a_n / (a_n + b_n):
            continue
        inits.append((a_n, b_n, a_s, b_s, 1 - w_signal, w_signal))
    # Random extras
    while len(inits) < n_init:
        mu_n = rng.uniform(0.001, max(0.02, median_r2))
        mu_s = rng.uniform(max(0.03, median_r2 * 2),
                            max(0.10, float(np.quantile(r2, 0.99))))
        nu_n = rng.uniform(50, 800)
        nu_s = rng.uniform(20, 200)
        a_n_free, b_n_free = mu_n * nu_n, (1 - mu_n) * nu_n
        a_n, b_n = _build_noise(a_n_free, b_n_free, mu_lo=mu_n)
        a_s, b_s = mu_s * nu_s, (1 - mu_s) * nu_s
        w_s = rng.choice([0.01, 0.02, 0.05, 0.10])
        inits.append((a_n, b_n, a_s, b_s, 1 - w_s, w_s))

    n_obs = len(r2)
    best = None
    for a_n0, b_n0, a_s0, b_s0, w_n0, w_s0 in inits:
        a_n, b_n = a_n0, b_n0
        a_s, b_s = a_s0, b_s0
        w_n, w_s = w_n0, w_s0
        prev_ll = -np.inf
        for _ in range(max_iter):
            log_p_n = beta_dist.logpdf(r2, a_n, b_n) + np.log(w_n + 1e-300)
            log_p_s = beta_dist.logpdf(r2, a_s, b_s) + np.log(w_s + 1e-300)
            m = np.maximum(log_p_n, log_p_s)
            log_norm = m + np.log(np.exp(log_p_n - m) + np.exp(log_p_s - m))
            ll = float(log_norm.sum())
            resp_n = np.exp(log_p_n - log_norm)
            resp_s = 1.0 - resp_n
            n_n = resp_n.sum()
            n_s = n_obs - n_n
            w_n = float(np.clip(n_n / n_obs, 1e-4, 1 - 1e-4))
            w_s = 1.0 - w_n
            new_n = _weighted_beta_mom(resp_n, r2)
            new_s = _weighted_beta_mom(resp_s, r2)
            if new_n is None or new_s is None:
                break
            if fixed_alpha_n is None:
                a_n, b_n = new_n
            else:
                mu_n_post = (resp_n * r2).sum() / max(n_n, 1e-9)
                mu_n_post = float(np.clip(mu_n_post, 1e-6, 0.999))
                a_n = fixed_alpha_n
                b_n = a_n * (1 - mu_n_post) / mu_n_post
            a_s, b_s = new_s
            # Enforce signal mean > noise mean — swapping breaks the d1
            # constraint, so reject the restart instead.
            if a_s / (a_s + b_s) <= a_n / (a_n + b_n):
                if fixed_alpha_n is not None:
                    a_n = b_n = a_s = b_s = float('nan')
                    break
                a_n, a_s = a_s, a_n
                b_n, b_s = b_s, b_n
                w_n, w_s = w_s, w_n
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll
        if not (np.isfinite(a_n) and np.isfinite(b_n)
                and np.isfinite(a_s) and np.isfinite(b_s)):
            continue
        mean_n = a_n / (a_n + b_n)
        mean_s = a_s / (a_s + b_s)
        if mean_s - mean_n < 0.001:
            continue
        if best is None or ll > best['ll']:
            best = {
                'mixture':        'f_beta',
                'noise_alpha':    float(a_n),
                'noise_beta':     float(b_n),
                'noise_dof1':     float(2 * a_n),
                'noise_dof2':     float(2 * b_n),
                'noise_d1_fixed': (float(d1_noise) if d1_noise is not None
                                   else None),
                'noise_mean_r2':  float(mean_n),
                'noise_weight':   float(w_n),
                'signal_alpha':   float(a_s),
                'signal_beta':    float(b_s),
                'signal_mean_r2': float(mean_s),
                'signal_weight':  float(w_s),
                'log_likelihood': ll,
                'n_voxels':       int(n_obs),
                'll':             ll,
            }
    if best is None:
        raise ValueError("all restarts failed to fit a non-trivial mixture")
    return best


def r2_fdr_threshold_f_beta(fit, alpha=0.05, n_grid=4000):
    """Tail-FDR R² threshold for the F-noise / Beta-signal mixture.

    Returns ``np.inf`` when no threshold in (0, 1) achieves the requested α.
    """
    from scipy.stats import beta as beta_dist
    grid = np.linspace(1e-6, 1 - 1e-6, n_grid)
    sf_n = 1.0 - beta_dist.cdf(grid, fit['noise_alpha'], fit['noise_beta'])
    sf_s = 1.0 - beta_dist.cdf(grid, fit['signal_alpha'], fit['signal_beta'])
    denom = fit['noise_weight'] * sf_n + fit['signal_weight'] * sf_s
    fdr = np.where(denom > 1e-12,
                   fit['noise_weight'] * sf_n / np.maximum(denom, 1e-12),
                   1.0)
    hits = np.where(fdr <= alpha)[0]
    if len(hits) == 0:
        return float('inf')
    return float(grid[hits[0]])


def plot_r2_f_beta_mixture(fit, r2=None, alpha=None, threshold=None, ax=None,
                            title=None):
    """Diagnostic plot for :func:`fit_r2_f_beta_mixture` (raw R² scale)."""
    import matplotlib.pyplot as plt
    from scipy.stats import beta as beta_dist
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    if r2 is not None:
        r2_clean = np.asarray(r2, dtype=float).ravel()
        r2_clean = r2_clean[np.isfinite(r2_clean) & (r2_clean > 0) & (r2_clean < 0.99)]
        upper = max(0.1, float(np.percentile(r2_clean, 99.5)) * 1.1)
        ax.hist(r2_clean, bins=80, range=(0, upper), density=True,
                color='0.85', edgecolor='0.5', alpha=0.9,
                label=f"data (n={len(r2_clean)})")
    else:
        upper = max(0.2, fit['signal_mean_r2'] * 3)
    grid = np.linspace(1e-5, upper, 600)
    p_n = (fit['noise_weight']
           * beta_dist.pdf(grid, fit['noise_alpha'], fit['noise_beta']))
    p_s = (fit['signal_weight']
           * beta_dist.pdf(grid, fit['signal_alpha'], fit['signal_beta']))
    ax.plot(grid, p_n, color='#1f77b4', lw=2,
            label=f"Noise F({fit['noise_dof1']:.1f}, {fit['noise_dof2']:.1f}), "
                  f"w={fit['noise_weight']:.3f}")
    ax.plot(grid, p_s, color='#d62728', lw=2,
            label=f"Signal Beta({fit['signal_alpha']:.1f}, {fit['signal_beta']:.1f}), "
                  f"w={fit['signal_weight']:.3f}")
    ax.plot(grid, p_n + p_s, color='k', lw=1, ls='--', alpha=0.6)
    if threshold is None and alpha is not None:
        threshold = r2_fdr_threshold_f_beta(fit, alpha=alpha)
    if threshold is not None and np.isfinite(threshold):
        label = f"Threshold R²={threshold:.3f}"
        if alpha is not None:
            label += f" (α={alpha})"
        ax.axvline(threshold, color='k', ls=':', lw=1.4, label=label)
    ax.set_xlim(0, upper)
    ax.set_xlabel("R²")
    ax.set_ylabel("Density")
    ax.legend(loc='upper right', fontsize=9)
    if title is not None:
        ax.set_title(title)
    return fig


def get_map(p):
    stimuli = p.columns.to_frame(index=False).T
    return stimuli.groupby(level=0).apply(lambda d: (p * d.values).sum(1) / p.sum(1)).T

def get_rsq(data, predictions, zerovartonan=True, allow_biased_residuals=False):

    resid = data - predictions

    # Pandas .sum() defaults to skipna=True, which silently drops NaN
    # rows from the sum. When model predictions are NaN (e.g. σ → 0 in
    # softplus collapses the Gaussian / DoG / DN PRF), this returns
    # ssq_resid = 0 and gives the infamous "phantom R² = 1" voxels.
    # Force skipna=False so NaN predictions propagate to NaN R², which
    # downstream code can filter explicitly.
    def _sumna(x):
        try:
            return x.sum(0, skipna=False)
        except TypeError:  # numpy fallback (no skipna kwarg)
            return x.sum(0)

    ssq_data = _sumna((data - data.mean(0)) ** 2)
    if allow_biased_residuals:
        ssq_resid = _sumna((resid - resid.mean(0)) ** 2)
    else:
        ssq_resid = _sumna(resid ** 2)

    r2 = (1 - (ssq_resid / ssq_data))

    if zerovartonan:
        r2[data.var() == 0] = np.nan

    r2.name = 'r2'

    return r2


def get_r(data, predictions):

    data_ = data - data.mean(0)
    predictions_ = predictions - predictions.mean(0)

    r = (data_*predictions_ ).sum(0)
    r = r / (np.sqrt((data_**2).sum(0) * (predictions_**2).sum(0)))

    return r
