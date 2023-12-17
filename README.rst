.. Braincoder documentation master file, created by
   sphinx-quickstart on Tue Nov 21 10:10:09 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Braincoder's documentation!
======================================

**Braincoder** is a package to fit encoding models to neural data (for now fMRI) and
to then *invert* those model to decode stimulus information from neural data.

Important links
===============

- Official source code repo: https://github.com/Gilles86/braincoder/tree/main
- HTML documentation (stable release): https://braincoder-devs.github.io/


Installation
============

Note that you need a environment with both `tensorflow-probability` and
`tensorflow`.

Set up miniforge
-----------------

(Only do this if you don't have conda installed)
I reccomend to use `miniforge <https://github.com/conda-forge/miniforge>`_,
make sure you use the ``mamba``-solver and set ``channel-priority`` to ``strict``:

.. code-block:: bash

        # Install mamba solver and set channel priority
        conda install mamba -n base -c conda-forge
        conda config --set channel_priority strict.


Install braincoder
------------------

Here we create a new environment called `braincoder` with the required packages:

.. code-block:: bash

    mamba create --name braincoder tensorflow-probability tensorflow -c conda-forge
    mamba activate braincoder
    pip install git+https://github.com/Gilles86/braincoder.git

Usage
=====

Please have a look at the `tutorials <https://braincoder-devs.github.io/tutorial/index.html>`_ to get started.
