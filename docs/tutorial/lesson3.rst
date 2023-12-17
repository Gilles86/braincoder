.. _tutorial_lesson3:

====================================================================
Lesson 3: Building the likelihood (by fitting the covariance matrix)
====================================================================

For many neuroscientific questions, we are interested in the relationship between neural codes
and objective stimulus featuers. For example, we might want to know how the brain represents
numbers, orientatoins, or spatial positions, and how these representatios change as a function
of task demands, attention, or prior expectations.

One particular powerful approach is to *decode* stimulus features from neural data
in a Bayesian fashion (e.g., van Bergen et al., 2015; Baretto-Garcia et al, 2023).

Inverting the encoding models
#############################

Here, we show how we go from a **determistic** forward model (i.e., a model that predicts neural
responses from stimulus features) to a **probabilistic** inverse model (i.e., a model that 
predicts stimulus features from neural responses). We will do so using a **Bayesian inversion scheme**:

.. math::
    p(s | x; \theta) = \frac{p(x | s; \theta) p(s)}{p(x)}

where :math:`s` is a n-dimensional point in stimulus space, and :math:`x` is a n-dimensional
activation pattern in neural space, and :math:`p(s | x; \theta)` is the posterior probability of
stimulus :math:`s` given neural response :math:`x` and model parameters :math:`\theta`.

A multivariate likelihood function
##################################

The crucial element that is still lacking for this Bayesian inversion scheme is a **likelihood function**.
Note stanard encoding models do not have a likelihood function, because they are deterministic
(:math:`f(x;\theta): s \mapsto x`).
They give us the average neural respons to a certain stimulus, but they don't tell us how likely a certain 
neural response is, given a certain stimulus.
However, we can easily derive a likelihood function from the forward model by adding Gaussian noise:

.. math::
    p(x | s; \theta) = f(s; \theta) + \epsilon

where :math:`\epsilon` is a *multivariate Normal distribution*

.. math::
    \epsilon \sim \mathcal{N}(0, \Sigma)


Estimating :math:`\Sigma`
#########################

The problem with high-dimensional covariance matrices
*****************************************************

How to set the covariance matrix :math:`\Sigma`? 
One approach would be to assume independent noise across neural dimensions (e.g.,
fMRI voxels), and use a spherical covariance matrix :math:`\Sigma = \tau^T\tau I`, where
:math:`\tau` is a vector containing the standard deviation of the residuals of the encoding models
and I is the identity matrix. 
However, if there *is* substantial covariance between the noise terms of different neural
dimensions (i.e., voxels), this could have severe consequences for the decoding performance.
In particular, the posterior might be overly confident in its predictions, assuming
independt sources of information that are not. Under some circumstances, the mean posterior,
the point estimate of the decoded stimulus, can also be affected.
Van Bergen et al. (2015, 2017) showed that modeling some of the covariance is ineed crucial
for making correct inferences about neural data.
However, we generally have a large number of voxels and very limited data.
Therefore, estimating the full covariance
matrix is not feasible (note that the number of free parameters scales quadratically with
the number of voxels :math:`p= n \times \frac{n-1}{2}`). Note that this is a general
problem of estimating covariance, and not specific to our use case (e.g., Ledoit, ...).

Van Bergen et al. (2015, 2017) proposed a two-part solution.

Regularizing the covariance matrix
**********************************

The first proposal of van Bergen et al. (2015, 2017), based on the work of
Ledoit (...), is to use a *shrinkage estimator* to estimate the covariance matrix.
Specifically, the free parameter :math:`\rho` scales between a perfectly-correlated
covariance matrix (:math:`\rho = 1`) and a diagonal covariance matrix (:math:`\rho = 0`):

.. math::
    \Sigma = \rho \tau^T\tau + (1-\rho) \tau^T\tau I

Accounting for shared noise due to shared neural populations
************************************************************

Van Bergen et al. (2015) also note that *voxels that share more tuned neural populations*
should also be more highly correlated. They do so by adding a second term to the covariance
matrix, which is based on the similarity of the tuning curves of the voxels,
given by the weight matrix :math:`W`.
Recall that W is a :math:`m \times n` matrix, where :math:`n` is the number of voxels,
and :math:`m` is the number of neural populations. So :math:`WW^T` is a :math:`n \times n`
matrix that contains the similarity between all pairs of voxels. We scale this
matrix by the free parameters :math:`\sigma^2`:

.. math::
    \sigma^2 WW^T

.. note::
    You might realize now that non-linear encoding models, like a standard
    ``GaussianPRFModel`` does not have a weight matrix :math:`W`. There are
    two work arounds here: 
    * Use the identity matrix :math:`I` as weight-matrix. This corresponds to saying that all neural populations in all the voxels are unique and share no noise sources.
    * Rewrite the individual encoding functions :math:`f(\theta_{1..i})` to linear functions that can be interpreted as a weight matrix :math:`W`. For PRF models, we can just set up a grid of plausible stimulus values and set the height of the PRF as the weight of those stimulus values. The weight matrix :math:`W` effectively then describes the amount of overlap between the receptive fields of different voxels. The noise covariance is then assumed to scale with the amount of overlap in recpeptive fields, which seems quite plausible and has worked well in earlier work (e.g., Baretto-Garcia et al., 2023).

The complete formula for the covariance matrix is thus:

.. math::
    \Sigma = \rho \tau^T\tau + (1-\rho) \tau^T\tau I  + \sigma^2 W^TW

Thus, the :math:`n \times n`covariance-matrix :math:`\Sigma` is now described by 
:math:`n + 2` parameters (the :math:`\tau` noise vector of length
:math:`n` plus :math:`\rho` and :math:`\sigma`.

Using the sample covariance
***************************

Note that additional elements can be added to the covariance matrix.
For one, we can add a proportion :math:`\lambda` of the empirical 
noise covariance matrix :math:`S` to the regularized covariance matrix, to allow for
a more sophisticated noise model:

.. math::
    \Sigma' = (1-\lambda) \cdot \Sigma + \lambda S

Using the anatomical distance
*****************************
Similarly, one could add a term that accounts for the physical
distance between different neural sources (i.e., voxels).


Fitting :math:`\Sigma`
##########################

``braincoder`` contains the ``ResidualFitter``-class that can be used
to fit a noise covariance matrix, and thereby a likelihood function
to a fitted ``EncodingModel``.

Here we first set up a simple ``VonmisesPRF``-model and simulate some data
with covarying noise.

.. literalinclude:: /../examples/00_encodingdecoding/fit_residuals.py
    :start-after: # We set up a simple VonMisesPRF model
    :end-before: # %%

Now we can import the `ResidualFitter` and estimate :math:`\Sigma` (here called
`omega` for legacy reasons):

.. include:: ../auto_examples/00_encodingdecoding/fit_residuals.rst
    :start-after: .. GENERATED FROM PYTHON SOURCE LINES 41-53
    :end-before: .. rst-class:: sphx-glr-timing


.. note::
    Here we have used the *generating* parameters and weights to fit the
    ``omega``-matrix. Note that in real data, we would use estimated parameters
    and/or weights.

.. note::

    The complete Python script and its output can be found :ref:`here<sphx_glr_auto_examples_00_encodingdecoding_fit_residuals.py>`.

Summary
#######

In this lesson we have seen:
 * We need to add a noise model to the classical encoding models :math:`f(s, \theta): s \mapsto x` to get a **likelihood function** which we can invert.
 * Conctretely, we add a multivariate Gaussian noise model to the deterministic predictions of the encodig models
 * We need a noise covariance matrix :math:`\Sigma` (or ``omega``) for this to work.
 * We use a regularised estimate of the covariance matrix.

 In the:ref:`next lesson<tutorial_lesson4>`, we will further explore how we can use a likelihood
 function to map from neural responses to stimulus features.