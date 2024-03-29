import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
import pandas as pd
from timeit import default_timer as timer

def cleanup_chain(chain, name, time_index):

    n_chains = chain.shape[1]
    n_samples = chain.shape[0]

    chain_index = pd.Index(range(n_chains), name='chain')

    if not issubclass(type(time_index), pd.core.indexes.base.Index):
        time_index = pd.Index(frames, name='frame')
    else:
        time_index = time_index
        
    sample_index = pd.Index(range(n_samples), name='sample')

    chain = [pd.DataFrame(chain[:, ix, :], index=sample_index,
                          columns=time_index) for ix in range(n_chains)]
    chain = pd.concat(chain, keys=chain_index)

    return chain.stack(list(range(chain.columns.nlevels))).to_frame(name).sort_index()

@tf.function
def sample_hmc(
        init_state,
        step_size,
        target_log_prob_fn,
        unconstraining_bijectors,
        target_accept_prob=0.85,
        unrolled_leapfrog_steps=1,
        max_tree_depth=10,
        num_steps=50,
        burnin=50):


    def trace_fn(_, pkr):
        return {
            'log_prob': pkr.inner_results.inner_results.target_log_prob,
            'diverging': pkr.inner_results.inner_results.has_divergence,
            'is_accepted': pkr.inner_results.inner_results.is_accepted,
            'accept_ratio': tf.exp(pkr.inner_results.inner_results.log_accept_ratio),
            'leapfrogs_taken': pkr.inner_results.inner_results.leapfrogs_taken,
            'step_size': pkr.inner_results.inner_results.step_size}

    hmc = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn,
        unrolled_leapfrog_steps=unrolled_leapfrog_steps,
        max_tree_depth=max_tree_depth,
        step_size=step_size)

    hmc = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=hmc,
        bijector=unconstraining_bijectors)

    adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=hmc,
        num_adaptation_steps=int(0.8 * burnin),
        target_accept_prob=target_accept_prob,
        # NUTS inside of a TTK requires custom getter/setter functions.
        step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
            inner_results=pkr.inner_results._replace(
                step_size=new_step_size)
        ),
        step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
        log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
    )
    
    start = timer()
    # Sampling from the chain.
    samples, stats = tfp.mcmc.sample_chain(
        num_results=burnin + num_steps,
        current_state=init_state,
        kernel=adaptive_sampler,
        trace_fn=trace_fn)

    duration = timer() - start
    stats['elapsed_time'] = duration

    return samples, stats



class Periodic(tfb.Bijector):

    def __init__(self, low, high, validate_args=False, name='periodic'):

        self.low = low
        self.high = high
        self.width = high - low

        super(Periodic, self).__init__(
            is_constant_jacobian=True,
            validate_args=validate_args,
            forward_min_event_ndims=0,
            name=name)

    def _forward(self, x):
      return ((x - self.low) % self.width) + self.low

    def _inverse(self, y):
      return y

    # def _inverse_log_det_jacobian(self, y):
      # return -self._forward_log_det_jacobian(self._inverse(y))

    # def _forward_log_det_jacobian(self, x):
      # # The full log jacobian determinant would be tf.zero_like(x).
      # # However, we circumvent materializing that, since the jacobian
      # # calculation is input independent, and we specify it for one input.
      # return tf.constant(0., x.dtype)
