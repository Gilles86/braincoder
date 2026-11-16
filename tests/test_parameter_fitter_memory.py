"""``ParameterFitter.fit`` must not keep autograd graphs between iterations.

On the torch backend, the R² tracking inside the fit loop used to run with
autograd on, and ``best_r2`` / ``best_parameters`` kept that graph alive. The
retained graph grew at every iteration until the GPU ran out of memory.
"""
from __future__ import annotations

import gc

import keras
import numpy as np
import pandas as pd
import pytest

from helpers import Bound

from braincoder.models import GaussianPRF
from braincoder.optimize import ParameterFitter
from braincoder.optimize import parameter_fitter as pf


def _graph_nodes():
    """Autograd nodes reachable from every live tensor."""
    import torch
    seen = set()
    stack = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.grad_fn is not None:
                stack.append(obj.grad_fn)
        except Exception:
            continue
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        stack.extend(next_node for next_node, _ in node.next_functions)
    return len(seen)


@pytest.mark.skipif(keras.backend.backend() != 'torch', reason='torch autograd only')
def test_fit_does_not_grow_the_autograd_graph(monkeypatch):
    n_t, n_voxels = 60, 8
    paradigm = pd.DataFrame({'x': np.linspace(-5, 5, n_t, dtype=np.float32)})
    true_pars = pd.DataFrame({
        'mu': np.linspace(-3, 3, n_voxels).astype(np.float32),
        'sd': np.full(n_voxels, 1.0, dtype=np.float32),
        'amplitude': np.full(n_voxels, 2.0, dtype=np.float32),
        'baseline': np.zeros(n_voxels, dtype=np.float32),
    })
    model = Bound(GaussianPRF(), paradigm, true_pars)
    data = model.predict(paradigm=paradigm, parameters=true_pars)

    original = pf.compute_gradients
    nodes = []

    def counting(loss_fn, variables):
        gc.collect(); nodes.append(_graph_nodes())
        return original(loss_fn, variables)

    monkeypatch.setattr(pf, 'compute_gradients', counting)

    init = true_pars.copy()
    init['mu'] += .5
    ParameterFitter(model, data, paradigm).fit(
        init_pars=init, max_n_iterations=20, min_n_iterations=20,
        learning_rate=.01, progressbar=False)

    assert len(nodes) == 20
    assert nodes[-1] <= nodes[2], f'autograd graph grew during the fit: {nodes}'
