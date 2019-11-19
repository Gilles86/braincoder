import os.path as op
import os
from braincoder.models import VoxelwiseGaussianReceptiveFieldModel
from nilearn import input_data, image
import pandas as pd
import numpy as np
from braincoder.utils import get_rsq
import scipy.stats as ss

subject = 2
session = 1

sourcedata = op.join(os.environ['HOME'], 'data', 'value_prf', 'ds-value')
derivatives = op.join(sourcedata, 'derivatives')
smoothing = False
single_trial = False

n_runs = 8
progressbar = True

n_voxels = 100


ims = []
for run in range(n_runs):
    for trial in range(1, 5):
        beta = op.join(sourcedata, f'sub-{subject:05d}/ses-{session:05d}/func/model2/beta_{run*27+trial:04d}.nii')
        im = image.load_img(beta)
        ims.append(im)

if smoothing:
    print('smoothing')
    ims = [image.smooth_img(im, 5.0) for im in ims]

mask = op.join(derivatives, 'masks', 'MNI_BA14_manual.nii')

mask_resampled = image.resample_to_img(mask, im, interpolation='nearest')
masker = input_data.NiftiMasker(mask_img=mask_resampled)

all_betas = image.concat_imgs(ims)
betas = masker.fit_transform(all_betas)
betas = pd.DataFrame(betas)

if single_trial:
    betas['run'] = np.repeat(np.arange(1, 9), 16)
    betas['trial'] = list(range(1, 17)) * 8
    betas = betas.set_index(['run', 'trial'])
else:
    betas['run'] = np.repeat(np.arange(1, 9), 4)
    betas['value_bin'] = [1,2,3,4] * 8
    betas = betas.set_index(['run', 'value_bin'])

df = pd.read_csv(op.join(sourcedata, 'ratingsMat.csv'), sep=',').set_index(['subjectID', 'runNr'])
df = df.loc[subject]

if single_trial:
    df = betas.set_index(df['Val'], append=True)
else:
    df['value_bin'] = df.groupby('runNr').Val.apply(lambda x: pd.qcut(x, 4, labels=[1,2,3,4]))
    values = df.groupby(['runNr', 'value_bin']).mean()[['Val']] / 900.
    values.index.rename(['run', 'value_bin'], inplace=True)
    df = values.merge(betas, left_index=True, right_index=True).set_index(['Val'], append=True)

model = VoxelwiseGaussianReceptiveFieldModel()

results = []

for test_run in range(1, 9):

    train = df.drop(test_run).copy()
    test = df.loc[test_run].copy()

    train_values = train.index.get_level_values('Val').astype(np.float32)
    test_values = test.index.get_level_values('Val').astype(np.float32)

    costs, pars, pred = model.fit_parameters(train_values, train,
                                             patience=100, progressbar=progressbar)

    r2 = get_rsq(df, pred)
    mask = r2.sort_values(ascending=False).iloc[:n_voxels].index
    model.apply_mask(mask)

    bins = np.linspace(0, 1, 150, endpoint=True)
    bin_model = model.to_bin_model(bins)

    bin_model.fit_residuals(train_values, train.loc[:, mask], lambd=lambd)

    pdf, map_, sd, (lower_ci, higher_ci) = bin_model.get_stimulus_posterior(test.loc[:, mask],
                                                                            stimulus_range=bins,
                                                                            normalize=True)

    r = pd.concat((pdf, map_), axis=1, keys=['pdf', 'pars'])

    print(f'RUN {run}, r={ss.pearsonr(map_, test_values):.02f}')

    results.append(r)

results = pd.concat(results)

target_dir = op.join(derivatives, 'decoding_prf{smoothing_ext}', f'sub-{subject}', 'func')

if not op.exists(target_dir):
    os.makedirs(target_dir)
    
results.to_pickle(op.join(target_dir, f'sub-{subject}_lambd-{lambd}_nvoxels-{n_voxels}_pdf.pkl'))
