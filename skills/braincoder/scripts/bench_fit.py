"""Time braincoder ParameterFitter (Adam GD) per iteration across model
classes, voxel counts and timepoints, on whatever device TF sees.

    python bench_fit.py --out bench.tsv [--n-voxels 500 2000 10000 50000]

Per configuration: fit(10) to trace and warm up, then fit(20) and fit(120)
from the same init; seconds/iteration = (t120 - t20) / 100, which cancels
tracing and per-call setup. ``first_10_iter_s`` is the warm-up call (tracing
cost). Random binary paradigm on a 50 x 50 grid (2500 pixels), fixed SPM
HRF unless the model name ends in '+flexHRF'. OOM and other errors are
recorded, not raised.
"""
import argparse
import os
import platform
import socket
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from braincoder.hrf import SPMHRFModel
from braincoder.models import (DifferenceOfGaussiansPRF2DWithHRF,
                               DivisiveNormalizationGaussianPRF2DWithHRF,
                               GaussianPRF2DWithHRF)
from braincoder.optimize import ParameterFitter

MODELS = {
    'Gaussian': (GaussianPRF2DWithHRF, dict(baseline=0., amplitude=1.), False),
    'DoG': (DifferenceOfGaussiansPRF2DWithHRF,
            dict(baseline=0., amplitude=1., srf_amplitude=.2, srf_size=3.), False),
    'DoG+flexHRF': (DifferenceOfGaussiansPRF2DWithHRF,
                    dict(baseline=0., amplitude=1., srf_amplitude=.2, srf_size=3.), True),
    'DN': (DivisiveNormalizationGaussianPRF2DWithHRF,
           dict(rf_amplitude=1., srf_amplitude=.2, srf_size=3., neural_baseline=1.,
                surround_baseline=1., bold_baseline=0.), False),
}


def run(name, n_vox, n_t, rng):
    cls, extra, flex = MODELS[name]
    g = np.linspace(-5, 5, 50)
    grid = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2).astype(np.float32)
    paradigm = (rng.random((n_t, len(grid))) < .05).astype(np.float32)
    data = pd.DataFrame(rng.normal(size=(n_t, n_vox)).astype(np.float32))
    init = pd.DataFrame(dict(x=rng.uniform(-3, 3, n_vox), y=rng.uniform(-3, 3, n_vox),
                             sd=rng.uniform(.5, 2, n_vox), **{k: np.full(n_vox, v) for k, v in extra.items()}))
    model = cls(grid_coordinates=grid, paradigm=paradigm, hrf_model=SPMHRFModel(tr=1.6),
                flexible_hrf_parameters=flex)
    if flex:
        init['hrf_delay'], init['hrf_dispersion'] = 4.5, .75
    init = init[model.parameter_labels].astype(np.float32)
    fitter = ParameterFitter(model, data, paradigm)
    times = {}
    for n in (10, 20, 120):   # first call pays tracing; the difference of the other two is pure iterations
        t0 = time.time()
        fitter.fit(max_n_iterations=n, min_n_iterations=n, init_pars=init,
                   learning_rate=.01, progressbar=False)
        times[n] = time.time() - t0
    return (times[120] - times[20]) / 100, times[10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-voxels', type=int, nargs='+', default=[500, 2000, 10000, 50000])
    ap.add_argument('--n-timepoints', type=int, nargs='+', default=[774, 3096])
    ap.add_argument('--models', nargs='+', default=list(MODELS))
    a = ap.parse_args()
    gpus = tf.config.list_physical_devices('GPU')
    device = (tf.config.experimental.get_device_details(gpus[0]).get('device_name', 'GPU')
              if gpus else f'CPU x{os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())}')
    rng = np.random.default_rng(0)
    rows = []
    for name in a.models:
        for n_t in a.n_timepoints:
            for n_vox in a.n_voxels:
                try:
                    s_it, first = run(name, n_vox, n_t, rng)
                    err = ''
                except Exception as e:      # OOM etc.: record and go on
                    s_it, first, err = np.nan, np.nan, type(e).__name__
                rows.append(dict(device=device, host=socket.gethostname(), model=name, n_timepoints=n_t,
                                 n_voxels=n_vox, s_per_iter=s_it, first_10_iter_s=first, error=err,
                                 tf=tf.__version__, cpu=platform.processor()))
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(a.out, sep='\t', index=False)


if __name__ == '__main__':
    main()
