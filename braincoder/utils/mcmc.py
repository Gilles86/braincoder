import pandas as pd


def cleanup_chain(chain, name, frames):

    n_chains = chain.shape[1]
    n_samples = chain.shape[0]

    chain_index = pd.Index(range(n_chains), name='chain')
    time_index = pd.Index(frames, name='frame')
    sample_index = pd.Index(range(n_samples), name='sample')

    chain = [pd.DataFrame(chain[:, ix, :], index=sample_index,
                          columns=time_index) for ix in range(n_chains)]
    chain = pd.concat(chain, keys=chain_index)

    return chain.stack('frame').to_frame(name).reorder_levels(['sample', 'chain', 'frame']).sort_index()
