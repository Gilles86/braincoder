.. _tutorial_lesson5:

====================================================
Lesson 5: Decoding noisy neural data
====================================================

In this lesson, we will see a more elaborate example of how we 
can decode stimulus features from neural data and check how well it worked.

Simulate data
#############


First we simulate data from a ``GaussianPRF`` with 20 PRFs
a 50 time points. The parameters are randomly generated
within a plausible range.

.. literalinclude:: /../examples/00_encodingdecoding/decode.py
    :start-after: # Set up a neural model
    :end-before: # %% 


Estimate parameters of encoding model :math:`\theta`
####################################################

Now we can use ``ParameterFitter`` to estimate back the parameters.
We also print out the average correlation between the true and estimated
parameters, for each parameter seperately.

.. include:: ../auto_examples/00_encodingdecoding/decode.rst
    :start-after: .. GENERATED FROM PYTHON SOURCE LINES 34-53
    :end-before: .. GENERATED FROM PYTHON SOURCE LINES 54-62

Note how some parameters (e.g., ``mu`` and ``amplitude``) are easier
to recover than others (e.g., ``sd``).

Estimate the covariance matrix :math:`\Sigma`
#############################################

To convert the **encoding model** into a **likelihood model**,
we need to add a **noise model**. More specifically, a 
Gaussian noise model. This requires us to estimate 
the covariance matrix :math:`\Sigma` of the noise.

Note how we use ``init_pseudoWWT`` and a plausible ``stimulus_range``
to approximate the ``WWT`` matrix.

.. literalinclude:: /../examples/00_encodingdecoding/decode.py
    :start-after: # Now we fit the covariance matrix
    :end-before: # %% 

Decode stimuli
##############

Simulate "unseen" data
**********************

Now we have estimated the **encoding model** and the **noise model**,
we can use them to decode the stimuli from the neural data.
For that, we simulate a new set of neural data and decode the stimuli
from.

.. literalinclude:: /../examples/00_encodingdecoding/decode.py
    :start-after: # Now we simulate unseen test data:
    :end-before: # And decode the test paradigm

Calculate likelihood density for a range of stimuli
***************************************************

We use ``model.get_stimulus_pdf`` to get the likelihood
for a plausible range of stimuli.

.. literalinclude:: /../examples/00_encodingdecoding/decode.py
    :start-after: # And decode the test paradigm
    :end-before: # %%

Plot posterior distributions
****************************

Here, we plot some of the posterior distributions for the stimuli.

.. literalinclude:: /../examples/00_encodingdecoding/decode.py
    :start-after: # Finally, we make some plots to see how well the decoder did
    :end-before: # %%

.. |posteriors| image:: ../auto_examples/00_encodingdecoding/images/sphx_glr_decode_001.png
.. centered:: |posteriors|

Estimate the mean posterior
********************************************

We can also estimate the mean posterior.
To do this we should just take the expectation of the posterior,
which is an integral:

.. math::
    \mathbb{E}[s] = \int s \cdot p(s|x) ds

similarly, we can also calculate the expected distance of the
expected stimulus :math:``E[s]`` to the true stimulus:

.. math::
    \mathbb{E}[d] = \int \|s - \mathbb{E}[s]\| \cdot p(s|x) ds

.. literalinclude:: /../examples/00_encodingdecoding/decode.py
    :start-after: # Let's look at the summary statistics of the posteriors posteriors
    :end-before: # Let's see how far the posterior mean is from the ground truth

Once we have these two summary statistics of the posterior, we can compare
them to the ground truth.

First of all, how close is the MAP stimulus to the true stimulus?

.. literalinclude:: /../examples/00_encodingdecoding/decode.py
    :start-after: # Let's see how far the posterior mean is from the ground truth
    :end-before: # Let's see how the error depends on the standard deviation of the posterior


.. |decoding_error| image:: ../auto_examples/00_encodingdecoding/images/sphx_glr_decode_002.png
.. centered:: |decoding_error|

That looks pretty good!

Finally, is the standard deviation of the posterior a good estimate
of the true noise level?

.. literalinclude:: /../examples/00_encodingdecoding/decode.py
    :start-after: # Let's see how the error depends on the standard deviation of the posterior
    :end-before: # %%

.. |decoding_error_error| image:: ../auto_examples/00_encodingdecoding/images/sphx_glr_decode_003.png
.. centered:: |decoding_error_error|

That looks pretty good too! The standard deviation of the posterior is actually
predictive of the true noise level.

Maximum a posteriori (MAP) estimate
####################################

Alternatively, instead of the mean of the posterior, we can also try to 
find the maximum a posteriori (MAP) estimate. This is the stimulus that
has the highest probability of being the true stimulus, given the neural data.

The MAP estimate is the stimulus that maximizes the posterior probability:

.. math::
    \hat{s} = \arg\max_s p(s|x)

Note that there are theoretical advantages to the **mean posterior** over the **MAP estimate**.
However, when the stimulus is high-dimensional, it is often easier to find the MAP estimate.

``braincoder`` has a ``StimulusFitter``, that can findo MAP estimates of the stimulus for you.
Stimulus optimisation tends to be very sensitive to local minima, so we first 
use a grid search to find a good starting point for the optimisation.

.. literalinclude:: /../examples/00_encodingdecoding/decode.py
    :start-after: # Now, let's try to find the MAP estimate using gradient descent
    :end-before: # %%

Now we apply gradient descent.

.. literalinclude:: /../examples/00_encodingdecoding/decode.py
    :start-after: # We can then refine the estimate using gradient descent
    :end-before: # Let's see how well we did

And then plot how well we did.

.. literalinclude:: /../examples/00_encodingdecoding/decode.py
    :start-after: # Let's see how well we did
    :end-before: # %%

.. |estimates_vs_groundtruth| image:: ../auto_examples/00_encodingdecoding/images/sphx_glr_decode_004.png
.. centered:: |estimates_vs_groundtruth|


Note how in this example, the MAP and mean posterior estimates are almost identical.
However, the mean posterior, where we take into account uncertainty in all stimulus
posterior dimensions, has theoretical advantages, in particular when the stimulus dimensionality
is high (here it is only 1-dimensional).

.. note::

    The complete Python script and its output can be found :ref:`here<sphx_glr_auto_examples_00_encodingdecoding_decode.py>`.

Summary
#######
In this tutorial, we have seen how to decode stimuli from neural data
in a more realistic setting.

The concrete steps were:
 * Fit the parameters of the encoding model
 * Estimate the covariance matrix of the noise
 * Apply the likelihood model to get a posterior over stimuli
 * Use numerical integration to get the expected stimulus :math:`E[s]`
 * Use numerical integration to get the expected distance between the real and expected stimulus (the standard deviation of the posterior)
 * Use grid+gradient descent optimisation to get the most likely stimulus

In the :ref:`next tutorial <tutorial_lesson6>`, we will see how to do the same thing, but in a two-dimensional 
stimulus space!