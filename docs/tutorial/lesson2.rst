
.. _tutorial_lesson2:

================================
Lesson 2: Linear encoding models
================================

A linear encoding approach
###########################

In the previous lesson, we saw how we can define (non-linear) encoding models
and fit their parameter :math:`\theta` to predict voxel responses :math:`x`
using non-linear descent.

It is important to point out that in most of the literature, the parameters
:math:`\theta` are **asssumed to be fixed**. In such work, researchers assume
a limited set of :math:`m` neural populations, each with their own set of parameters
:math:`\theta`. 

The responses in different :math:`n` neural measures (e.g., fMRI voxels) are then to 
assume to be a **linear combination** of these fixed neural populations.
How much each neural population contributes to each voxel is then defined by a
weight matrix :math:`W` of size :math:`m \times n`.

.. math::
    x_j = \sum W_{i, j} \cdot f_j(\theta_j)

The big advantage of this approach is that it allows us fit the weight matrix
:math:`W` using linear regression. This is a much, much faster approach than fitting
the parameters :math:`\theta` using non-linear gradient descent.

In this lesson, we will see how we can use the ``EncodingModel`` class to fit
linear encoding models.

Setting up a linear encoding model in ``braincoder``
####################################################

In braincoder, we can set up a linear encoding model by defining a 
**fixed number of neural encoding populations**, each with their own
parameters set :math:`\theta_j`.

Here we use a Von Mises tuning curve to define the neural populations
that are senstive to the orientation of a grating stimulus. 
Note that orientations are given in radians, so lie between -pi and pi.

Set up a von Mises model
************************

.. literalinclude:: /../examples/00_encodingdecoding/linear_encoding_model.py
    :start-after: # Import necessary libraries
    :end-before: # %%

Once we have set up the model, we can first have a look at the predictions for the 
6 different basis functions:

.. literalinclude:: /../examples/00_encodingdecoding/linear_encoding_model.py
    :start-after: # Plot the basis functions
    :end-before: # %% 

.. |basis_functions| image:: ../auto_examples/00_encodingdecoding/images/sphx_glr_linear_encoding_model_001.png
.. centered:: |basis_functions|

Because the model also has a :math:`n{\times}m` weight matrix (number of voxels x number of neural populations),
we can also use the model to predict the responses of different voxels to the same orientation stimuli:

.. literalinclude:: /../examples/00_encodingdecoding/linear_encoding_model.py
    :start-after: # Plot the predicted responses for the 3 voxels
    :end-before: # %% 

.. |voxel_predictions1| image:: ../auto_examples/00_encodingdecoding/images/sphx_glr_linear_encoding_model_002.png
.. centered:: |voxel_predictions1|

Fit a linear encoding model using (regularised) OLS
****************************************************

To fit linear encoding models we can use the ``braincoder.optimize.WeightFitter``.
This fits weights to the model using linear regression. Note that one can 
also provide an ``alpha``-parameter to the ``WeightFitter`` to regularize the
weights (pull them to 0; equivalent to putting a Gaussian prior on the weights). 
This is often a very good idea in real data!

.. literalinclude:: /../examples/00_encodingdecoding/linear_encoding_model.py
    :start-after: # Import the weight fitter
    :end-before: # %% 

.. |voxel_predictions2| image:: ../auto_examples/00_encodingdecoding/images/sphx_glr_linear_encoding_model_003.png
.. centered:: |voxel_predictions2|

.. note::

    The complete Python script and its output can be found :ref:`here<sphx_glr_auto_examples_00_encodingdecoding_linear_encoding_model.py>`.

Summary
#######
In this lesson, we had a look at *linear* encoding models.
These models
 * Assume a fixed number of neural populations, each with their own parameters :math:`\theta_j`
 * Every voxel then is assumed to be a linear combination of these neural populations
 * The weights of this linear combination can be fit using linear regression
 * This is much faster than fitting the parameters :math:`\theta_j` using non-linear gradient descent

In the :ref:`next lesson<tutorial_lesson3>`, we will see how we can add a *noise model*
to the encoding models, which yields a likelihood function which we can invert in 
a principled manner.