# Neural Encoding Models in TensorFlow

This work is highly inspired by van Bergen et al. (2015) and reimplments
that model and the underlying generative models using TensorFlow.
This allows for easy calculation of the gradients of the parameters 
wrt the likelihood.

This leads to model fits that are orders of magnitude faster than implementations
that use standard numpy or matlab, even when "only" CPUs are used.
