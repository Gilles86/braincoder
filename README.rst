.. Braincoder documentation master file, created by
   sphinx-quickstart on Tue Nov 21 10:10:09 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Braincoder's documentation!
======================================


**Braincoder** is a package to fit encoding models to neural data (for now fMRI) and
to then *invert* those model to decode stimulus information from neural data.

Usage
=====

Important links
---------------

- Official source code repo: https://github.com/Gilles86/braincoder/tree/main
- HTML documentation (stable release): https://braincoder-devs.github.io/


Installation
------------
Note that you need a environment with both `tensorflow-probability` and
`tensorflow`.

I reccomend to use `miniforge <https://github.com/conda-forge/miniforge>`_,
make sure you use the ``mamba``-solver and set ``channel-priority``!

.. code-block:: bash

        # Install mamba solver and set channel priority
        conda install mamba -n base -c conda-forge
        conda config --set channel_priority strict.


Here we create a new environment called `braincoder` with the required packages:

.. code-block:: bash

    conda create --name braincoder tensorflow-probability tensorflow -c conda-forge