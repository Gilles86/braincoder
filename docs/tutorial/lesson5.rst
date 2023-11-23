
.. _tutorial_lesson5:

====================================================
Lesson 5: Decoding noisy neural data
====================================================

In this lesson, we will see how we can decode actual data
and check how well it did.

Simulate data
#############

.. literalinclude:: /../examples/00_encodingdecoding/decode.py
    :start-after: # Set up a neural model
    :end-before: # %% 


Estimate parameters of encoding model :math:`\theta`
####################################################

.. literalinclude:: /../examples/00_encodingdecoding/decode.py
    :start-after: # Now we fit back the PRF parameters
    :end-before: # %% 

Estimate the covariance matrix :math:`\Sigma`
#############################################

.. literalinclude:: /../examples/00_encodingdecoding/decode.py
    :start-after: # Now we fit the covariance matrix
    :end-before: # %% 