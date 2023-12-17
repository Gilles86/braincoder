
.. _tutorial_lesson6:

====================================================
Lesson 6: Decoding two-dimensional stimulus spaces
====================================================

In this lesson, we will decode a **two-dimensional** stimulus 
space. Specifically, we are going to assume that the 
*stimulus drive* (`amplitude`) of specific stimuli can be modulated,
for example by the *attentional state* of the subject
(see, e.g., :footcite:t:`serences2007spatially`, :footcite:t:`sprague2013attention`, and
:footcite:t:`sawetsuttipan2023perceptual`).

Hence, every stimulus is now characterized by two parameters:
its *orientation* (``x (radians)``) and its *amplitude* (``amplitude``).

Let's set up a virtual **mapping experiment**, where we are simulating data, 
estimate parameters, and decode the stimulus space.

*Crucially*, in the mapping experiment, we are ('rightfully') going
to assume that the stimulus drive (``amplitude``) is not modulated
and is always ``1.0``.

We use the argument ``model_stimulus_amplitude`` of the ``VonMisesPRF``-model
to indicate we want to model both orientation and amplitudes.

.. include:: ../auto_examples/00_encodingdecoding/decode2d.rst
    :start-after: .. GENERATED FROM PYTHON SOURCE LINES 12-36
    :end-before: .. GENERATED FROM PYTHON SOURCE LINES 37-52

Now we simulate data, estimate parameters, and the noise model:

.. literalinclude:: /../examples/00_encodingdecoding/decode2d.py
    :start-after: # Now we can simulate some data and estimate parameters+noise
    :end-before: # %% 

After fitting the mapping paradigm, we are now going to simulate an 
experiment with two conditions, where the stimulus drive is modulated
in the **attended** condition, the stimulus drive is modulated by a factor of **``1.5``**,
whereas in the **unattended** condition, the stimulus drive is modulated by a factor of **``0.5``**.

.. literalinclude:: /../examples/00_encodingdecoding/decode2d.py
    :start-at: # Now we set up an experimental paradigm with two conditions
    :end-before: # %%

If we plot the simulated responses as a function of the ground truth orientation,
we can clearly see both the preferred orientation and the modulation of the stimulus drive
in the attended condition:

.. literalinclude:: /../examples/00_encodingdecoding/decode2d.py
    :start-at: # Plot the data
    :end-before: # %%

.. |data| image:: ../auto_examples/00_encodingdecoding/images/sphx_glr_decode2d_001.png
.. centered:: |data|

Now we have the data and the noise model we can, for each data point, calculate its likelihood
given different plausible stimuli. Note that this set of plausible stimuli 
consistutes a (flat) prior, so after normalizing likelihoods, we can interpret 
the likelihood as a posterior distribution. 
Also, remember that we use the PRF and noise parameters that we estimated
in the mapping experiment!

Note how we use the ``pandas`` library to flexibly manipulate the multidimensional likelihood;

.. literalinclude:: /../examples/00_encodingdecoding/decode2d.py
    :start-at: # Now we can calculate the 2D likelihood/posterior of different orientations+amplitudes for the data
    :end-before: # %%

Here we plot the posterior distribution for the first 9 data points of the **attended**
and **unattended** conditions:

.. literalinclude:: /../examples/00_encodingdecoding/decode2d.py
    :start-at: # Plot 2D posteriors for first 9 trials
    :end-before: # %%

.. |attended_posterior| image:: ../auto_examples/00_encodingdecoding/images/sphx_glr_decode2d_002.png
.. centered:: |attended_posterior|

.. |unattended_posterior| image:: ../auto_examples/00_encodingdecoding/images/sphx_glr_decode2d_003.png
.. centered:: |unattended_posterior|

These 2D posterior distributions are very insightful! Notice, for example:
 * The orientation of the attended stimulus is more precisely defined in the posterior than the unattended stimulus.
 * The amplitude of the attended stimulus is more precisely defined in the posterior than the unattended stimulus.
 * The MAP estimate/mean posterior (red ``+``) are closer to the ground truth for the attended stimulus than the unattended stimulus.

============================================
Decode the marginal probability distribution
============================================

When we now want to decode the marginal probability distribution of ``orientation`` 
we need to take into the account of the uncertainty of both dimensions. Hence,
we take the expectation, integrating the other dimension away:

.. math::
    p(\text{orientation}) = \int_{\text{amplitude}} p(\text{orientation}, \text{amplitude}) da

.. literalinclude:: /../examples/00_encodingdecoding/decode2d.py
    :start-at: # Now we can calculate the 1D posterior for specific orientations _or_ amplitudes
    :end-before: # %%

.. |orientation_posterior| image:: ../auto_examples/00_encodingdecoding/images/sphx_glr_decode2d_004.png
.. centered:: |orientation_posterior|

===================================================
Decode the conditional probability distribution
===================================================

We see that especially the orientation posterior of the unattended condition is very broad.
Part of the problem is that we have to take into account the uncertainty surrounding the
generating amplitude.
We can make the posterior more precise by using a ground truth amplitude, which we can
use to condition the posterior on:

.. math::
    p(\text{orientation} | \text{amplitude} = a) = p(\text{orientation}, \text{amplitude} = a)

(With :math:`a` being the ground truth amplitude).

.. literalinclude:: /../examples/00_encodingdecoding/decode2d.py
    :start-at: # Use the ground truth amplitude to improve the orientation posterior
    :end-before: # %%

.. |orientation_posterior_conditional| image:: ../auto_examples/00_encodingdecoding/images/sphx_glr_decode2d_005.png
.. centered:: |orientation_posterior_conditional|

Note how especially for the unattended condition, the orientation posterior is now much more precise.

=================
The complex plane
=================

If we want to have calculate the *mean posterior* for ``orientation`` and ``amplitude``,
we can use numerical integration. Notice, however, that ``orientation`` is in polar space,
which means we can not just integrate over raw angles (e.g., what is the mean angle of ``[0.1*pi, 1.9*pi]``?).

The problem is solved by using integration over ``complex`r numbers. 
Briefly, complex numbers represent a point in the complex plane, where the real part is the ``x``-coordinate
and the imaginary part is the ``y``-coordinate. 

This is what the complex plane looks like for the first 10 data points of the **attended** condition:

.. literalinclude:: /../examples/00_encodingdecoding/decode2d.py
    :start-at: # Let's plot the firs 10 trials in the complex plane
    :end-before: # %%

.. |10trials_complex_plane| image:: ../auto_examples/00_encodingdecoding/images/sphx_glr_decode2d_006.png
.. centered:: |10trials_complex_plane|

If we now want to take integrals/expectations/averages over angles, we can simply
take the average over the complex numbers that represent the angles. which
is equivalent to taking the average over the unit circle in the complex plane.

.. include:: ../auto_examples/00_encodingdecoding/decode2d.rst
    :start-after: .. GENERATED FROM PYTHON SOURCE LINES 217-254
    :end-before: .. GENERATED FROM PYTHON SOURCE LINES 255-289

Note how the correlation between the ground truth and posterior mean
is much higher for the attended condition than the unattended condition.

Also note that you should never use normal (e.g., Pearson's) correlation
on angles!

We can now also plot the mean posteriors in the complex plane:

.. literalinclude:: /../examples/00_encodingdecoding/decode2d.py
    :start-at: # Let's see how far the posterior mean is from the ground truth
    :end-before: # %%

.. |estimates_complex_plane| image:: ../auto_examples/00_encodingdecoding/images/sphx_glr_decode2d_007.png
.. centered:: |estimates_complex_plane|

Note how the unattended condition is, indeed, generally much further away from the ground truth
than the attended condition.

Note, also that even **within** conditions we can predict the error of the classifier
by using the variance of the posterior distribution:

.. literalinclude:: /../examples/00_encodingdecoding/decode2d.py
    :start-at: # Plot the error as a function of the standard deviation of the posterior
    :end-before: # %%

.. |error_vs_posterior_sd| image:: ../auto_examples/00_encodingdecoding/images/sphx_glr_decode2d_008.png
.. centered:: |error_vs_posterior_sd|

References
==========
.. footbibliography::
