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
        raise ValueError(
            f"need ≥50 finite R² values in (0, 0.99); got {len(r2)}")
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
        'mixture':        'gmm_logit',
        'noise_mu':       float(means[n_idx]),
        'noise_sigma':    float(sds[n_idx]),
        'noise_weight':   float(ws[n_idx]),
        'noise_mean_r2':  float(_inv_logit(means[n_idx])),
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


def r2_posterior_signal(r2, fit):
    """Posterior P(signal | r²) from a 2-component mixture fit.

    Per-voxel responsibility of the *signal* (higher-mean) component:

        P(signal | z) = w_s · N(z | μ_s, σ_s²) /
                        [w_n · N(z | μ_n, σ_n²) + w_s · N(z | μ_s, σ_s²)]

    where ``z = logit(R²)``. Values outside the open interval
    ``(0, 1)`` map to 0 (cannot be signal under the logit-Gaussian
    mixture).

    Args:
        r2: array-like of per-voxel R² values.
        fit: dict returned by :func:`fit_r2_mixture`.

    Returns:
        ``np.ndarray`` of P(signal | r²) values aligned with ``r2``.
    """
    from scipy.stats import norm
    r2 = np.asarray(r2, dtype=float).ravel()
    out = np.zeros_like(r2)
    valid = np.isfinite(r2) & (r2 > 0) & (r2 < 1)
    if not valid.any():
        return out
    r2_safe = np.clip(r2[valid], 1e-6, 1 - 1e-6)
    z = _logit(r2_safe)
    p_n = fit['noise_weight'] * norm.pdf(z, fit['noise_mu'],
                                          fit['noise_sigma'])
    p_s = fit['signal_weight'] * norm.pdf(z, fit['signal_mu'],
                                           fit['signal_sigma'])
    denom = p_n + p_s
    out[valid] = np.where(denom > 0, p_s / np.maximum(denom, 1e-300), 0.0)
    return out


def r2_p_signal_threshold(r2_or_fit, p=0.5, n_grid=4000):
    """R² value at which P(signal | r²) first crosses ``p``.

    Searches a logit grid from below the noise mean to above the
    signal mean for the smallest r² with P(signal | r²) ≥ p. Returns
    ``np.inf`` if the posterior never reaches ``p`` on that grid
    (near-degenerate mixture).

    Accepts either an R² array (fits a fresh mixture via
    :func:`fit_r2_mixture`) or a previously-fitted dict.
    """
    if not isinstance(r2_or_fit, dict):
        fit = fit_r2_mixture(r2_or_fit)
    else:
        fit = r2_or_fit
    z_grid = np.linspace(fit['noise_mu'] - 5 * fit['noise_sigma'],
                          fit['signal_mu'] + 8 * fit['signal_sigma'],
                          n_grid)
    r2_grid = _inv_logit(z_grid)
    p_signal = r2_posterior_signal(r2_grid, fit)
    hits = np.where(p_signal >= p)[0]
    if len(hits) == 0:
        return float('inf')
    return float(r2_grid[hits[0]])


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
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=8)
    if title is not None:
        ax.set_title(title)
    return fig


def get_map(p):
    stimuli = p.columns.to_frame(index=False).T
    return stimuli.groupby(level=0).apply(lambda d: (p * d.values).sum(1) / p.sum(1)).T

def get_rsq(data, predictions, zerovartonan=True, allow_biased_residuals=False):

    resid = data - predictions

    # ssq_data = np.clip(((data - data.mean(0))**2).sum(0), 1e-5, None)
    ssq_data = ((data - data.mean(0))**2).sum(0)
    if allow_biased_residuals:
        ssq_resid = ((resid - resid.mean(0))**2).sum(0)
    else:
        ssq_resid = (resid**2).sum(0)

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
