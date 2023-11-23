.. _tutorial_lesson4:

====================================================
Lesson 4: From neural responses to stimulus features
====================================================

Now we have a likelihood function :math:`p(x|s;\theta)`, we know,
**for a given stimulus**, how likely different multivariate neural responses
are. What we might be interested in the inverse, :math:`p(s|x;\theta)`,
how likely different stimuli are *for a given BOLD pattern*. 

To get here, we can use Bayes rule:

.. math::
    p(s|x, \theta) = \frac{p(x|s, \theta) p(s)}{p(x)}

Note that
 * We need to define a prior :math:`p(s)`
 * To be able to integrate over :math:`p(s|x, \theta)` we need to approximate :math:`p(x)` by normalisation.
..  * For a given :math:`x`, we want to evaluate the likelihood at all possible :math:`s`

In practice, for simple one or two-dimensional stimulus space,
we can use a uniform prior and evaluate likelihoods at a grid of useful
points within that prior.

The crucial insight here is that we, for a given neural response pattern :math:`x`
**try out which of a large set of possible stimuli are actually consistent with this
neural response pattern**

Let's say we observed the following orientation receptive field, centered around
a half :math:`pi` and we observe, in unseen data, an activation of slightly less
than 0.2:

.. |rf_activation| image:: ../auto_examples/00_encodingdecoding/images/sphx_glr_invert_model_001.png
.. centered:: |rf_activation|

We can now set up a likelihood function and find which stimuli are consistent with
this activation pattern as follows.

First set up the model..

.. literalinclude:: /../examples/00_encodingdecoding/invert_model.py
    :start-after: # Set up six evenly spaced von Mises PRFs
    :end-before: # %% 

Then evaluate the likelihood for different orientations

.. literalinclude:: /../examples/00_encodingdecoding/invert_model.py
    :start-after: # Evaluate the likelihood of different possible orientations
    :end-before: # %% 

.. |likelihood1| image:: ../auto_examples/00_encodingdecoding/images/sphx_glr_invert_model_002.png
.. centered:: |likelihood1|

You can see that the likelihood is highest around :math:`\frac{1}{8}\pi` and
:math:`\frac{3}{8}\pi`! With only one receptive field the predictions for these
two points in stimulus space are identical!


If we have two RFs, the situation becomes unambiguous:

.. |rf_activation2| image:: ../auto_examples/00_encodingdecoding/images/sphx_glr_invert_model_003.png
.. centered:: |rf_activation2|


.. literalinclude:: /../examples/00_encodingdecoding/invert_model.py
    :start-after: # Set up 2-dimensional model to invert
    :end-before: # %% 

.. |likelihood2| image:: ../auto_examples/00_encodingdecoding/images/sphx_glr_invert_model_004.png
.. centered:: |likelihood2|


Summary
#######

We have seen how to set up a simple encoding model and how to invert it to
see which stimulis :math:`s` are consistent with a given neural response :math:`x`.

For real data we of course have hundreds of voxels and thousands of stimuli.
In the :ref:`next tutorial <tutorial_lesson5>` we will see how we can
decode data in a more natural setting.
