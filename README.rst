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

.. _installation:

Installation
------------
Note that you need a environment with both `tensorflow-probability` and
`tensorflow`.

I reccomend to use conda (with `strict channel priority on conda-forge <https://conda-forge.org/docs/user/tipsandtricks.html#how-to-fix-it>` 
and `mamba <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`):
.. code-block:: bash

   conda create --name braincoder tensorflow-probability tensorflow -c conda-forge


.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. toctree::
   :hidden:
   :includehidden:
   :titlesonly:

   auto_examples/index.rst
