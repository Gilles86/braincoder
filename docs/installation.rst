Installation
============

Braincoder requires Python 3.10+ and `Keras 3 <https://keras.io>`_ with at least one
backend (TensorFlow, JAX, or PyTorch).

Recommended: conda + editable install
--------------------------------------

This approach gives you a full development environment with TensorFlow as the backend:

.. code-block:: bash

    git clone https://github.com/Gilles86/braincoder.git
    cd braincoder
    conda env create -f environment.yml
    conda activate braincoder

The ``environment.yml`` installs TensorFlow, Keras, and braincoder itself in editable
mode (``pip install -e .``), so local changes to the source are reflected immediately.

Choosing a backend
------------------

Keras 3 supports three backends. Set the ``KERAS_BACKEND`` environment variable before
importing braincoder:

.. code-block:: bash

    export KERAS_BACKEND=tensorflow  # default
    export KERAS_BACKEND=jax
    export KERAS_BACKEND=torch

Or set it permanently in ``~/.keras/keras.json``:

.. code-block:: json

    {"backend": "tensorflow"}

Each backend must be installed separately. The ``environment.yml`` includes all three
(``tensorflow``, ``jax[cpu]``, ``torch``), but you can install only the one you need.

pip-only install
----------------

braincoder is not on PyPI, so install a tagged release from GitHub, together with a
backend:

.. code-block:: bash

    pip install "braincoder @ git+https://github.com/Gilles86/braincoder.git@v0.6.2" tensorflow   # TensorFlow
    # or
    pip install "braincoder @ git+https://github.com/Gilles86/braincoder.git@v0.6.2" "jax[cpu]"   # JAX
    # or
    pip install "braincoder @ git+https://github.com/Gilles86/braincoder.git@v0.6.2" torch        # PyTorch

Pin a tag like this in your project's environment, so that its results stay
reproducible.

For a development install from source:

.. code-block:: bash

    git clone https://github.com/Gilles86/braincoder.git
    cd braincoder
    pip install -e . tensorflow

Additional dependencies
-----------------------

Some features require optional packages:

- **MCMC sampling**: ``blackjax`` (JAX backend recommended)
- **Notebooks / tutorials**: ``jupyter``, ``ipykernel``
- **Tests**: ``pytest`` (or ``pip install -e ".[test]"`` from a clone)

Verifying the install
---------------------

.. code-block:: bash

    python -c "import braincoder; print(braincoder.__version__)"
    pytest tests/
