#!/usr/bin/env python
"""Print what a braincoder job will actually run on, and fail early if it is wrong.

Put it at the top of a job script, before the expensive step:

    export KERAS_BACKEND=tensorflow
    python preflight.py --backend tensorflow --require-gpu --min-version 0.6.1

Exits 1 on a mismatch. Catches the failures a "does TF see the GPU" check misses:
~/.keras/keras.json selecting another backend, a CPU-only jaxlib on a GPU node, and an
import that resolves to a different braincoder checkout than expected.
"""
import argparse
import os
import sys


def devices(backend):
    """Devices the *active Keras backend* can use, not what TF alone reports."""
    if backend == 'tensorflow':
        import tensorflow as tf
        return [d.name for d in tf.config.list_physical_devices('GPU')]
    if backend == 'jax':
        import jax
        return [str(d) for d in jax.devices() if d.platform != 'cpu']
    if backend == 'torch':
        import torch
        if torch.cuda.is_available():
            return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        return ['mps'] if torch.backends.mps.is_available() else []
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--backend', choices=['tensorflow', 'jax', 'torch'],
                    help='Fail unless Keras uses this backend.')
    ap.add_argument('--require-gpu', action='store_true', help='Fail unless that backend sees a GPU.')
    ap.add_argument('--min-version', help='Fail if braincoder is older, e.g. 0.6.1.')
    ap.add_argument('--expect-path', help='Fail unless braincoder imports from under this path.')
    a = ap.parse_args()

    problems = []
    env_backend = os.environ.get('KERAS_BACKEND')
    if a.backend and env_backend != a.backend:
        problems.append(f'KERAS_BACKEND={env_backend!r}; export it as {a.backend!r} so keras.json cannot decide')

    try:
        import keras
    except ImportError as e:
        sys.exit(f'FAIL: cannot import keras ({e})')
    major = int(keras.__version__.split('.')[0])
    backend = keras.backend.backend()
    if major < 3:
        problems.append(f'keras {keras.__version__}: this env can only run a pre-port braincoder')

    import braincoder
    version = getattr(braincoder, '__version__', 'unknown')
    path = os.path.dirname(os.path.abspath(braincoder.__file__))
    found = devices(backend)

    print(f'braincoder {version} from {path}')
    print(f'keras {keras.__version__}, backend {backend} (KERAS_BACKEND={env_backend})')
    print(f'accelerators visible to {backend}: {found or "none"}')

    if a.backend and backend != a.backend:
        problems.append(f'active backend is {backend}, expected {a.backend}')
    if a.require_gpu and not found:
        problems.append(f'{backend} sees no GPU; the job would silently run on CPU')
    if a.expect_path and not path.startswith(os.path.abspath(os.path.expanduser(a.expect_path))):
        problems.append(f'braincoder imported from {path}, expected under {a.expect_path}')
    if a.min_version:
        as_tuple = lambda v: tuple(int(p) for p in v.split('.')[:3])
        try:
            if as_tuple(version) < as_tuple(a.min_version):
                problems.append(f'braincoder {version} is older than {a.min_version}')
        except ValueError:
            problems.append(f'cannot compare version {version!r} with {a.min_version!r}')

    if problems:
        for p in problems:
            print(f'FAIL: {p}', file=sys.stderr)
        sys.exit(1)
    print('preflight OK')


if __name__ == '__main__':
    main()
