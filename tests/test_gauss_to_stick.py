from braincoder.models import VoxelwiseGaussianReceptiveFieldModel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

model = VoxelwiseGaussianReceptiveFieldModel()
palette = sns.color_palette()

n_voxels = 25
n_timepoints =  150

noise = 1.0

paradigm = np.tile(np.arange(0, 20), int(n_timepoints / 20 + 1))
paradigm = paradigm[:n_timepoints]

parameters = np.ones((n_voxels, 4))
parameters[:, 0] = np.linspace(0, 20, n_voxels)
parameters[:, 1] = np.abs(np.random.randn(n_voxels)) * 3
# parameters[:, 3] = np.random.randn(n_voxels)

data = model.simulate(paradigm, parameters, noise=noise)


costs, pars_, pred_ =  model.fit_parameters(paradigm, data, progressbar=True)
stimuli = np.linspace(-20, 40, 1000)
sm = model.to_stickmodel(basis_stimuli=stimuli)

sm.fit_residuals(data=data)

data2 = model.simulate(paradigm, parameters, noise=noise)
s, map, sd, ci = sm.get_stimulus_posterior(data2, stimulus_range=stimuli, normalize=True)
plt.plot(paradigm, color=palette[0])
plt.plot(map, ls='--', color=palette[1])
plt.title('r = {:.2f}'.format(ss.pearsonr(map.ravel(), paradigm)[0]))
plt.fill_between(range(len(map)), ci[0][:, 0], ci[1][:, 0],
        alpha=0.2, color=palette[1])

plt.figure()
# s = np.clip(s, np.percentile(s, 1), np.percentile(s, 99))
sns.heatmap(s)
plt.show()
plt.figure()
plt.plot(stimuli, s[:5].T)
plt.show()
