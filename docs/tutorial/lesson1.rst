.. _tutorial_lesson1:

============================
Lesson 1: Encoding models
============================

Encoding models
###############

Standard encoding models, like 2D PRF models that are used 
in retinotopic mapping, but also models of numerosity
and Gabor orientations define **determinsitic one-to-one mapping** from 
**stimulus space** to **BOLD response space**:

.. math::

   f(x; \theta): s \mapsto x


Here:
 * The stimulus :math:`s` is some point in a n-dimensional feature space. For example, :math:`s` could be the *numerosity* of a stimulus array, the *orientation* of a Gabor patch, or a *2D image*.
 * :math:`x` is a BOLD activation pattern *in a single voxel*. This could both be a single-trial estimate, or the actual activity pattern on a given timepoint :math:`t`
 * :math:`\theta` is a set of parameters tha define the particular mapping of a particular voxel. For example, the center and dispersion of a 2D PRF, or the preferred orientation of a Gabor patch. In some cases, :math:`\theta` is fitted to data. In other cases, it is fixed.

Concrete example
****************


One standard encoding model is the **1D Gaussian PRF**. This is simply the probability density
of a 1D Gaussian distribution, centered at :math:`\mu` and with dispersion :math:`\sigma`, evaluated at
:math:`x`, multiplied by an amplitude :math:`a`: and added to a baseline :math:`b`:

.. math::
   f(x; \mu, \sigma, a, b) = a \cdot \mathcal{N}(x; \mu, \sigma) + b


Simulate data
*************

This model is implemented in the ``GaussianPRF`` class in ``braincoder.models``:

.. literalinclude:: /../examples/00_encodingdecoding/encoding_model.py
    :start-after: # Import necessary libraries
    :end-before: # %%


.. |prf_timeseries| image:: ../auto_examples/00_encodingdecoding/images/sphx_glr_encoding_model_001.png
.. centered:: |prf_timeseries|


We can also simulate noisy data and try to estimate back the generating parameters:

.. literalinclude:: /../examples/00_encodingdecoding/encoding_model.py
    :start-after: # We simulate data with a bit of noise
    :end-before: # %%
    
.. |noisy_prf_timeseries| image:: ../auto_examples/00_encodingdecoding/images/sphx_glr_encoding_model_002.png
.. centered:: |noisy_prf_timeseries|

Estimate parameters
*******************

Then we can try to estimate back the generating parameters using a grid search.
This code automatically picks, for each voxel, the parameters that maximize the 
**correlation** between the predicted and actual BOLD response:

.. literalinclude:: /../examples/00_encodingdecoding/encoding_model.py
    :start-after: # Import and set up a parameter fitter with the (simulated) data,
    :end-before: # %%


The grid search only optimized for the center and dispersion of the Gaussian,
but we also want to optimize for the amplitude and baseline,
using ordinary least squares (note that this is computationally much cheaper than adding
``amplitude`` and ``baseline`` to the grid search).
Now we minimize the sum of squared errors between the predicted and actual BOLD response
**R2**. 

.. literalinclude:: /../examples/00_encodingdecoding/encoding_model.py
    :start-after: # We can now fit the amplitude and baseline using OLS
    :end-before: # %%

This grid-optimized parameters already fit the data pretty well.
Note how we can use ``get_rsq`` to calculate the *fraction of explained variance R2*
between the predicted and actual BOLD response. This is the exact same R2 that is used
to optimize the parameters on.

.. |grid_fit| image:: ../auto_examples/00_encodingdecoding/images/sphx_glr_encoding_model_003.png
.. centered:: |grid_fit|

We can do even better using *gradient descent* optimisation. Note that because `braincoder` uses 
`tensorflow`, it uses autodiff-calculated exact gradients, as well as the GPU to speed up the
computation. This is especially useful for more complex models. `braincoder` is under some
circumstances multiple orders of magnitude faster than other PRF libraries.

.. literalinclude:: /../examples/00_encodingdecoding/encoding_model.py
    :start-after: # Final optimisation using gradient descent:
    :end-before: # %%

.. |gd_fit| image:: ../auto_examples/00_encodingdecoding/images/sphx_glr_encoding_model_004.png
.. centered:: |gd_fit|

.. note::

    The complete Python script its output can be found :ref:`here<sphx_glr_auto_examples_00_encodingdecoding_encoding_model.py>`.

Summary
#######

In this lesson we have seen:
 * Encoding models define a deterministic mapping from stimulus space to BOLD response space
 * `braincoder` allows us to define all kins of encoding models, and to fit them to data
 * `braincoder` uses tensorflow to speed up the computation, and to allow for gradient descent optimisation
 * Fitting encoding models usually take the following steps:
   * Set up a grid of possible non-linear parameter values, and find best-fitting ones (``optimizer.fit_grid()``)
   * Fit linear parameters using ordinary least squares (``optimizer.refine_amplitude_baseline()``)
   * Finialize fit using gradient descent (``optimizer.fit()``

In the :ref:`next lesson<tutorial_lesson2>`, we will see how we can fit *linear* encoding models.
