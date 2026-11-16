# Models and parameters

## Canonical equations

All predictions are convolved with the HRF after the neural response, then `bold_baseline` (or `baseline`) is added.

| Model | Neural response per TR |
|---|---|
| Gaussian | `amplitude · ⟨S, G(σ)⟩ + baseline` |
| DoG | `amplitude · ⟨S, G(σ)⟩ − amplitude · srf_amplitude · srf_size · ⟨S, G(σ·srf_size)⟩ + baseline` (with `rf_normalization='legacy'`) |
| Divisive normalization (Aqil et al. 2021; prfpy `Norm_Iso2DGaussianModel`) | `(A ⟨S,G1⟩ + B) / (C ⟨S,G2⟩ + D) − B/D` |

braincoder's DN parameters map onto Aqil's as follows, with `w(σ)` the scale factor from `rf_normalization` and prfpy's own `dx²` factor absorbed into A and C:

| Aqil | braincoder |
|---|---|
| A | `rf_amplitude · w(sd)` |
| B | `neural_baseline` |
| C | `srf_amplitude · \|rf_amplitude\| · w(sd · srf_size)` |
| D | `surround_baseline` |
| σ1, σ2 | `sd`, `sd · srf_size` |

Verified numerically on 2026-11-02: braincoder's DN equals the prfpy equation to float32 precision (max difference 2 × 10⁻⁷) for positive `rf_amplitude`. Two details differ by class and line (re-tested on 0.6.2):

- **`−B/D`:** the `*WithHRF` classes subtract it before convolution. The plain `DivisiveNormalizationGaussianPRF2D` returns `(A⟨S,G1⟩ + B) / (C⟨S,G2⟩ + D)`, so its rest level is `B/D`, not 0.
- **Negative `rf_amplitude`:** `rf_amplitude` has no positivity constraint. The denominator uses `|rf_amplitude|`, so it stays positive and negative pRFs are safe. Keras-line versions **before 0.6.2** used the signed amplitude: a negative amplitude lowered the denominator and could drive it through zero. Refit DN models from those versions if negative amplitudes occurred.

## `rf_normalization` (v0.6.1)

The scale of the Gaussian before the amplitude multiplies it.

| Value | Scale `w(σ)` | Meaning of amplitude | `srf_amplitude` (DN) is |
|---|---|---|---|
| `'legacy'` (default) | `pixel_area / (σ√(2π))`: a 1-D normalisation on a 2-D Gaussian | no clean meaning; scales with σ | surround:center total-drive ratio × `srf_size` |
| `'peak'` | 1 | drive per stimulated pixel at the pRF centre; same scale as prfpy's default | surround:center peak ratio |
| `'area'` | `pixel_area / (2πσ²)` | response when the whole pRF is stimulated; `⟨S,G⟩` is the stimulated fraction | surround:center total-drive ratio |

- The three are exact reparameterisations for a fixed σ: predictions match after converting the amplitudes by the ratio of scale factors (tested). The choice affects interpretation and conditioning, not the model family.
- Use `'peak'` for new DN fits. It is bounded as σ → 0 (`'area'` grows as 1/σ²), and the parameters are on prfpy's scale.
- `'legacy'` stays the default so existing fits and warm starts keep working. Convert legacy amplitudes with `A_peak = A_legacy · pixel_area / (σ√(2π))`, and the surround the same way with `σ·srf_size`.

## Transform bugs found so far

Both were silent: fits ran, R² looked plausible, no error. After adding a model or transform, always test the round trip and compare the prediction with a numpy implementation of the published equation.

| Class | Bug | Symptom | Fixed in |
|---|---|---|---|
| `DifferenceOfGaussiansPRF2D` | inherited the 5-column Gaussian transform and dropped `srf_amplitude`, `srf_size` | DoG R² worse than plain Gaussian; surround columns scrambled with HRF columns | TF line 2026-05-10 (7c50f2c); `main` v0.6.1 |
| `DivisiveNormalizationGaussianPRF2DWithHRF`, flexible HRF | forward transform passed `[bold_baseline, delay, dispersion]` to the HRF transform | `hrf_delay = 4·sigmoid(bold_baseline) + 3` exactly in every voxel; backward/forward does not round-trip | v0.6.1; TF line 2849288 |

Minimal test, for any class with a flexible HRF:

```python
back = model._transform_parameters_forward(model._transform_parameters_backward(pars))
np.testing.assert_allclose(back, pars, rtol=1e-4, atol=1e-4)
```

## Linearity shortcut

With σ's, pRFs and HRF fixed, a modulation-field model is linear in its gains, and any pRF model is linear in the stimulus. So:

- `X = (pred(g + h) − pred(g − h)) / (2h)` gives a regressor, and gains can be re-estimated with OLS in seconds rather than a nonlinear fit (per-run, per-condition, trial-wise).
- Multiply the paradigm by a per-TR modulator to get TR-resolved ramps (e.g. a learning curve within a block).
- Adding an offset δ changes the SSE by `ΔSSE(δ) = δ²·ΣX² − 2δ·ΣX·r`, with `r` the residual. Save `a = ΣX²` and `b = ΣX·r` per voxel, and any δ can be evaluated offline. The conditional optimum is `b/a`.
- **Always check it:** compare forward and backward differences, or reconstruct the full prediction. The modulation is rectified (`max(·, 0)`), so very negative gains break linearity. Exclude fits where the relative error is above 10⁻³; in practice that was about 1% of fits, all with gains at a bound.

## Modulation fields: identifiability

- A very wide field (σ at its upper bound) is a flat gain and trades off against amplitude; a very narrow one (below the item size) fits noise. Box the shared σ's (e.g. 0.5–8°) and gains (e.g. |g| ≤ 10).
- If a stimulus item is present on most trials, a flat field time-locked to it is nearly collinear with the item's own drive scale.
- "Gain" is signed in these models: below 1 is suppression. "Modulation" is the clearer word in figures.
