"""Post-fit validation helpers for PRF / encoding-model output.

After a ``ParameterFitter.fit`` produces a parameter DataFrame, these
helpers identify voxels whose fit isn't trustworthy (zero-variance
BOLD; NaN R² from a model blowup; etc.) and either flag them in place
or check that a *prior* fit can safely be re-used as warm-start init.

Everything operates on standard ``(T, V)`` BOLD arrays + per-voxel
parameter DataFrames; no project-specific structure assumed.
"""
from __future__ import annotations

import numpy as np
import warnings


def mask_valid_bold_voxels(data, var_threshold=1e-6):
    """Return a boolean mask (length V) marking voxels with
    non-degenerate BOLD signal.

    A voxel is *problematic* when its variance over time is at or below
    ``var_threshold``. Such voxels (dead, NaN-filled, drift-only, or
    entirely regressed-out by cleaning) cause fitters to emit sentinel
    rows.

    This helper does NOT filter the data — it just returns the mask.
    Use :func:`mark_invalid_fits` to apply it post-hoc to a fit-output
    DataFrame, which keeps the output shape aligned with the input
    voxel set (important for downstream NIfTI export).

    Args:
        data: (T, V) array of BOLD time courses.
        var_threshold: voxels with var(BOLD) <= this are flagged.

    Returns:
        (V,) bool ndarray.
    """
    return np.var(data, axis=0) > var_threshold


def mark_invalid_fits(pars, data, var_threshold=1e-6,
                      meta_cols=("subject", "hemi", "voxel_idx"),
                      verbose=True):
    """Post-hoc mark invalid voxel fits in-place on a parameter DataFrame.

    A voxel's fit is marked invalid when EITHER:

      - BOLD variance ≤ ``var_threshold`` (zero-variance / dead voxel
        — no signal to fit).
      - R² is NaN / non-finite (prediction blew up — e.g. DN
        denominator → 0 producing +Inf).

    For each invalid voxel:

      - all *parameter* columns are set to NaN,
      - the ``r2`` column is set to 0.0 (so the standard
        ``r² > FDR_thr`` selection drops it cleanly).

    Output shape is preserved — required for
    ``masker.inverse_transform(pars[col].values)`` to write NIfTIs.

    Args:
        pars: fit-output DataFrame; must have ``r2``.
        data: (T, V) BOLD matrix the fit was run on; rows of ``pars``
              correspond 1-to-1 with columns of ``data``.
        var_threshold: see :func:`mask_valid_bold_voxels`.
        meta_cols: columns that are NOT model parameters and should
                   not be NaN'd (e.g., subject ID).
        verbose: print a summary of the count flagged.

    Returns:
        the modified DataFrame (also modified in place).
    """
    if "r2" not in pars.columns:
        raise ValueError("pars must have an 'r2' column")
    valid_bold = mask_valid_bold_voxels(data, var_threshold=var_threshold)
    r2 = pars["r2"].to_numpy()
    invalid_r2 = ~np.isfinite(r2)
    invalid = (~valid_bold) | invalid_r2
    n_invalid = int(invalid.sum())
    if verbose:
        print(f"  mark_invalid_fits: flagged {n_invalid} of "
              f"{len(invalid)} voxels "
              f"(low BOLD var: {int((~valid_bold).sum())}, "
              f"non-finite R²: {int(invalid_r2.sum())})")
    if n_invalid > 0:
        param_cols = [c for c in pars.columns
                      if c != "r2" and c not in meta_cols]
        for col in param_cols:
            pars.loc[invalid, col] = np.nan
        pars.loc[invalid, "r2"] = 0.0
    return pars


def validate_prf_parameters(pars, *, sd_min=None, model_label=None,
                             source=None, raise_on_invalid=True):
    """Sanity-check PRF parameters loaded from disk before passing
    them into a fit.

    Catches contract violations EARLY (at load time) so the failure
    surfaces with caller context rather than as a deep braincoder
    assertion mid-GD. Checks:

      - ``sd <= 0``: mathematically impossible for a Gaussian PRF.
      - ``sd <= sd_min`` if ``sd_min`` is given: would trip braincoder's
        strict ``_sd_softplus_inverse``.
      - ``srf_size <= 1``: surround:center σ ratio must be > 1
        (surround wider than center); braincoder's DoG / DN transforms
        all enforce this bound.

    ``mark_invalid_fits`` sentinels (NaN params + r²=0) are not
    flagged — those rows are explicitly invalid and downstream code
    handles them.

    Args:
        pars: DataFrame with PRF parameter columns.
        sd_min: if given, σ floor that downstream fits will enforce.
        model_label, source: optional strings included in the error
                              message for caller context.
        raise_on_invalid: True → ``ValueError``. False → ``warnings.warn``.

    Returns:
        ``pars`` unchanged.

    Raises:
        ValueError: when ``raise_on_invalid`` and any issue is found.
    """
    issues = []
    if "r2" in pars.columns:
        active = pars["r2"].to_numpy() > 0
    else:
        active = np.ones(len(pars), dtype=bool)

    if "sd" in pars.columns:
        s = pars["sd"].to_numpy()
        finite_active = active & np.isfinite(s)
        n_nonpos = int((finite_active & (s <= 0)).sum())
        if n_nonpos:
            issues.append(
                f"  - {n_nonpos} voxels with sd <= 0 "
                f"(mathematically impossible for a Gaussian PRF)")
        if sd_min is not None and sd_min > 0:
            n_below = int((finite_active & (s > 0) & (s <= sd_min)).sum())
            if n_below:
                issues.append(
                    f"  - {n_below} voxels with sd <= sd_min "
                    f"(= {sd_min}). braincoder's _sd_softplus_inverse "
                    f"will raise on these. Likely cause: NIfTIs predate "
                    f"the sd_min hook — refit the source model with "
                    f"sd_min > 0 first")

    if "srf_size" in pars.columns:
        r = pars["srf_size"].to_numpy()
        finite_active = active & np.isfinite(r)
        n_below = int((finite_active & (r <= 1.0)).sum())
        if n_below:
            issues.append(
                f"  - {n_below} voxels with srf_size <= 1 "
                f"(surround should be wider than center). braincoder's "
                f"DoG / DN srf_size transform requires srf_size > 1; "
                f"likely cause: NIfTIs fit with the legacy DoG "
                f"transform that used sd_min as the srf_size floor — "
                f"refit the source DoG model")

    # Strictly-positive scalars in the DN parameterisation.  These
    # transforms are softplus-only, so an init value <= 0 would trip
    # softplus_inverse and produce NaN downstream.
    for pos_col in ("rf_amplitude", "neural_baseline",
                    "surround_baseline", "srf_amplitude"):
        if pos_col not in pars.columns:
            continue
        v = pars[pos_col].to_numpy()
        finite_active = active & np.isfinite(v)
        n_nonpos = int((finite_active & (v <= 0)).sum())
        if n_nonpos:
            issues.append(
                f"  - {n_nonpos} voxels with {pos_col} <= 0 "
                f"(braincoder's DN transform requires > 0; likely "
                f"cause: NIfTIs predate the positivity-floor commit)")

    if not issues:
        return pars

    msg = "Invalid PRF parameters"
    if model_label is not None:
        msg += f" (model {model_label})"
    if source is not None:
        msg += f" from {source}"
    msg += ":\n" + "\n".join(issues)

    if raise_on_invalid:
        raise ValueError(msg)
    warnings.warn(msg)
    return pars
