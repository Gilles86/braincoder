import scipy.stats as ss

def gaussian_normed(x, *pars):
    mu, sigma = pars

    # Normalize, so highest response is 1
    return ss.norm.pdf(x, mu, sigma) / ss.norm.pdf(mu, mu, sigma)
