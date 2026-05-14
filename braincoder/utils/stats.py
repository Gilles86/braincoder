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
