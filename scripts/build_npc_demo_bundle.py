"""Build a small demo bundle from /data/ds-neuralpriors for braincoder examples.

Picks the subject with highest mean NPCr CV-R² (model 15, smoothed,
whole-brain CV), then extracts a minimal self-contained package that the
example notebooks can drive end-to-end:

    r2_wholebrain.nii.gz           whole-brain within-sample R² (use for
                                   the mixture-model voxel selection)
    cv_r2_wholebrain.nii.gz        whole-brain cross-validated R² (for
                                   generalisation reporting / inspection)
    brain_mask.nii.gz              fmriprep EPI brain mask (T1w space)
    npcr/single_trial_betas.tsv.gz (n_trials × n_vox) single-trial GLM betas
    npcr/paradigm.tsv              per-trial stimulus (columns n, range)
    npcr/voxel_coords_mm.tsv       (n_vox × {x, y, z}) voxel centroids, T1w mm
    npcr/voxel_to_vertex.tsv       nearest-vertex idx per voxel (in patch)
    npcr/r2.tsv                    NPCr within-sample R² values
    npcr/cv_r2.tsv                 NPCr CV-R² values
    surface_patch.npz              cropped white-matter mesh (rh) around NPCr
    manifest.json                  subject id + provenance

The resulting zip is uploaded once to figshare; downstream loader fetches
and caches it.

Run in the `braincoder` conda env (only nilearn / nibabel / scipy / numpy /
pandas needed).
"""

from __future__ import annotations

import json
import os
import os.path as op
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.maskers import NiftiMasker
from scipy.spatial import cKDTree


BIDS = Path('/data/ds-neuralpriors')
DERIV = BIDS / 'derivatives'
MODEL = 15
SMOOTHED = True
PATCH_RADIUS_MM = 30.0  # crop surface patch within this mm of any matched vertex


def _model_dir(cv: bool) -> str:
    parts = [f'model{MODEL}']
    if cv:
        parts.append('cv')
    parts.append('whole_brain')
    if SMOOTHED:
        parts.append('smoothed')
    return '.'.join(parts)


def _wholebrain_cvr2_path(sub: str) -> Path:
    return (DERIV / 'encoding_models' / _model_dir(cv=True)
            / f'sub-{sub}' / 'func'
            / f'sub-{sub}_desc-cvr2.optim_space-T1w_pars.nii.gz')


def _wholebrain_r2_path(sub: str) -> Path:
    return (DERIV / 'encoding_models' / _model_dir(cv=False)
            / f'sub-{sub}' / 'func'
            / f'sub-{sub}_desc-r2.optim_space-T1w_pars.nii.gz')


def _npcr_mask_path(sub: str) -> Path:
    return (DERIV / 'ips_masks' / f'sub-{sub}' / 'anat'
            / f'sub-{sub}_space-T1w_desc-NPCr_mask.nii.gz')


def _single_trial_path(sub: str) -> Path:
    smoothed_part = '.smoothed' if SMOOTHED else ''
    return (DERIV / f'glm_stim1.denoise{smoothed_part}' / f'sub-{sub}' / 'func'
            / f'sub-{sub}_task-task_space-T1w_desc-stim_pe.nii.gz')


def _brain_mask_path(sub: str) -> Path:
    return (DERIV / 'fmriprep' / f'sub-{sub}' / 'ses-1' / 'func'
            / f'sub-{sub}_ses-1_task-task_run-2_space-T1w_desc-brain_mask.nii.gz')


def _white_surface_path(sub: str, hemi: str) -> Path:
    return (DERIV / 'fmriprep' / f'sub-{sub}' / 'anat'
            / f'sub-{sub}_hemi-{hemi}_white.surf.gii')


def _mu_wide_path(sub: str) -> Path:
    """Production fitted ``mu`` (preferred numerosity) for the wide range."""
    smoothed_part = '.smoothed' if SMOOTHED else ''
    return (DERIV / 'encoding_models' / f'model{MODEL}{smoothed_part}'
            / f'sub-{sub}' / 'func'
            / f'sub-{sub}_desc-mu.wide.optim_space-T1w_pars.nii.gz')


def _npcr_r2_path(sub: str) -> Path:
    """Production within-sample R² for the NPCr-restricted fit (same masker shape as mu)."""
    smoothed_part = '.smoothed' if SMOOTHED else ''
    return (DERIV / 'encoding_models' / f'model{MODEL}{smoothed_part}'
            / f'sub-{sub}' / 'func'
            / f'sub-{sub}_desc-r2.optim_space-T1w_pars.nii.gz')


def _event_path(sub: str, ses: int, run: int) -> Path:
    return (BIDS / f'sub-{sub}' / f'ses-{ses}' / 'func'
            / f'sub-{sub}_ses-{ses}_task-task_run-{run}_events.tsv')


def list_candidate_subjects() -> list[str]:
    """Subjects with all the artefacts the bundle needs."""
    out = []
    for d in sorted((DERIV / 'encoding_models' / _model_dir(cv=True)).glob('sub-*')):
        sub = d.name.replace('sub-', '')
        needed = [
            _wholebrain_cvr2_path(sub),
            _wholebrain_r2_path(sub),
            _npcr_mask_path(sub),
            _single_trial_path(sub),
            _brain_mask_path(sub),
            _mu_wide_path(sub),
            _npcr_r2_path(sub),
        ]
        if all(p.exists() for p in needed):
            out.append(sub)
    return out


def _squeeze_img(img):
    """Drop trailing singleton dims (the NPCr masks ship as 4D with one volume)."""
    data = np.asanyarray(img.dataobj)
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    return nib.Nifti1Image(data, img.affine, img.header)


def mean_npcr_cvr2(sub: str) -> float:
    """Mean CV-R² inside the NPCr mask (after resampling mask → cvr2 grid)."""
    cvr2_img = image.load_img(str(_wholebrain_cvr2_path(sub)), dtype=np.float32)
    mask_img = _squeeze_img(image.load_img(str(_npcr_mask_path(sub)), dtype=np.int32))
    mask_on_cvr2 = image.resample_to_img(
        mask_img, cvr2_img, interpolation='nearest',
        force_resample=True, copy_header=True,
    )
    cvr2 = cvr2_img.get_fdata()
    mask = mask_on_cvr2.get_fdata().astype(bool)
    if not mask.any():
        return float('-inf')
    return float(np.nanmean(cvr2[mask]))


STIM_LO, STIM_HI = 10.0, 40.0
R2_GOOD = 0.10  # cut-off for "trustworthy production fit" within NPCr


def score_subject_tuning_diversity(sub: str) -> tuple[float, dict[str, float]]:
    """Score how well a subject's NPCr tuning will demonstrate decoding.

    What we want for the tutorial is voxels whose *fitted preferred
    numerosity* sits broadly across the stimulus range [10, 40] — that
    way a Bayesian decoder applied to model-simulated responses can
    actually move with the stimulus. Subjects whose voxels mostly fit
    "monotonic ramp" solutions (mu < 10) or land outside the upper edge
    will show degenerate simulate+decode curves that collapse toward
    the mean of mu.

    Score = sqrt(N_good) · in_range_frac · sqrt(in_range_std + 0.1)
    where N_good is the number of NPCr voxels with production R² above
    ``R2_GOOD`` and a fitted mu in [10, 40], and in_range_frac /
    in_range_std are computed over those voxels.
    """
    mu_img = image.load_img(str(_mu_wide_path(sub)), dtype=np.float32)
    r2_img = image.load_img(str(_npcr_r2_path(sub)), dtype=np.float32)
    # mu/r2 are already in the NPCr-masker shape, but flatten safely.
    mu = np.asarray(mu_img.get_fdata(), dtype=np.float32).ravel()
    r2 = np.asarray(r2_img.get_fdata(), dtype=np.float32).ravel()
    valid = np.isfinite(mu) & np.isfinite(r2)
    good = valid & (r2 > R2_GOOD)
    n_good = int(good.sum())
    if n_good == 0:
        return float('-inf'), {'n_good': 0}
    mu_good = mu[good]
    in_range = (mu_good >= STIM_LO) & (mu_good <= STIM_HI)
    in_range_frac = float(in_range.mean())
    in_range_std = float(np.std(mu_good[in_range]) if in_range.any() else 0.0)
    score = float(np.sqrt(n_good) * in_range_frac * np.sqrt(in_range_std + 0.1))
    return score, {
        'n_good': n_good,
        'in_range_frac': in_range_frac,
        'in_range_std': in_range_std,
        'median_r2_npcr': float(np.median(r2[good])),
    }


def pick_best_subject() -> tuple[str, float, dict[str, dict]]:
    scores: dict[str, dict] = {}
    for sub in list_candidate_subjects():
        try:
            score, info = score_subject_tuning_diversity(sub)
            info['score'] = score
            scores[sub] = info
        except Exception as exc:  # noqa: BLE001
            print(f'  sub-{sub}: error → {exc!r}', file=sys.stderr)
    best = max(scores, key=lambda s: scores[s]['score'])
    return best, scores[best]['score'], scores


def load_paradigm(sub: str) -> pd.DataFrame:
    """Build a 480-row paradigm: columns n, range, indexed by trial order."""
    rows = []
    for ses in (1, 2):
        for run in range(1, 9):
            df = pd.read_csv(_event_path(sub, ses, run), sep='\t')
            stim = df[df['trial_type'] == 'stimulus'][['trial_nr', 'n']].copy()
            stim['range'] = 'wide' if (stim['n'] > 25).any() else 'narrow'
            stim['session'] = ses
            stim['run'] = run
            rows.append(stim)
    paradigm = pd.concat(rows, ignore_index=True)
    paradigm = paradigm[['session', 'run', 'trial_nr', 'n', 'range']]
    paradigm['n'] = paradigm['n'].astype(float)
    return paradigm.reset_index(drop=True)


def load_npcr_single_trials(sub: str) -> tuple[pd.DataFrame, np.ndarray, NiftiMasker]:
    """Return (betas, voxel_mm_coords, masker).

    betas: DataFrame (n_trials × n_vox) of single-trial GLM estimates within NPCr.
    voxel_mm_coords: (n_vox × 3) T1w mm.
    """
    mask_img = _squeeze_img(image.load_img(str(_npcr_mask_path(sub)), dtype=np.int32))
    pe_img = image.load_img(str(_single_trial_path(sub)), dtype=np.float32)
    # Resample mask onto functional grid.
    mask_on_func = image.resample_to_img(
        mask_img,
        image.index_img(pe_img, 0),
        interpolation='nearest',
        force_resample=True,
        copy_header=True,
    )
    masker = NiftiMasker(mask_img=mask_on_func)
    betas = masker.fit_transform(pe_img).astype(np.float32)  # (n_trials, n_vox)
    n_trials = betas.shape[0]
    if n_trials != 480:
        print(f'  warning: expected 480 trials, got {n_trials}', file=sys.stderr)

    # Voxel mm coordinates in T1w space.
    mask_arr = mask_on_func.get_fdata().astype(bool)
    affine = mask_on_func.affine
    ijk = np.argwhere(mask_arr)
    xyz = nib.affines.apply_affine(affine, ijk).astype(np.float32)

    betas_df = pd.DataFrame(betas, columns=pd.RangeIndex(betas.shape[1], name='voxel'))
    betas_df.index.name = 'trial'
    return betas_df, xyz, masker


def load_npcr_cvr2(sub: str, masker: NiftiMasker) -> np.ndarray:
    cvr2_img = image.load_img(str(_wholebrain_cvr2_path(sub)), dtype=np.float32)
    return masker.transform(cvr2_img).squeeze().astype(np.float32)


def load_npcr_r2(sub: str, masker: NiftiMasker) -> np.ndarray:
    r2_img = image.load_img(str(_wholebrain_r2_path(sub)), dtype=np.float32)
    return masker.transform(r2_img).squeeze().astype(np.float32)


def load_white_surface(sub: str, hemi: str) -> tuple[np.ndarray, np.ndarray]:
    gii = nib.load(str(_white_surface_path(sub, hemi)))
    vertices = gii.darrays[0].data.astype(np.float32)
    faces = gii.darrays[1].data.astype(np.int32)
    return vertices, faces


def crop_surface_patch(
    vertices: np.ndarray,
    faces: np.ndarray,
    seed_xyz: np.ndarray,
    radius_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop the mesh to vertices within ``radius_mm`` of any point in ``seed_xyz``.

    Returns (patch_vertices, patch_faces, old_to_new_index_map) where the map
    is -1 for vertices outside the patch.
    """
    tree = cKDTree(vertices)
    # All vertices that have *any* seed within radius_mm.
    keep_sets = tree.query_ball_point(seed_xyz, r=radius_mm)
    keep = np.zeros(len(vertices), dtype=bool)
    for s in keep_sets:
        keep[s] = True
    old_to_new = -np.ones(len(vertices), dtype=np.int32)
    old_to_new[keep] = np.arange(keep.sum(), dtype=np.int32)
    # Keep faces whose three vertices are all in the patch.
    face_mask = keep[faces].all(axis=1)
    patch_faces = old_to_new[faces[face_mask]]
    patch_vertices = vertices[keep]
    return patch_vertices, patch_faces, old_to_new


def match_voxels_to_vertices(
    voxel_xyz: np.ndarray,
    vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(vertices)
    dist, idx = tree.query(voxel_xyz)
    return idx.astype(np.int32), dist.astype(np.float32)


def build_bundle(sub: str, out_dir: Path) -> Path:
    print(f'[{sub}] assembling bundle in {out_dir}')
    out_dir.mkdir(parents=True, exist_ok=True)
    npcr_dir = out_dir / 'npcr'
    npcr_dir.mkdir(exist_ok=True)

    # --- whole-brain artefacts ------------------------------------------------
    shutil.copy(_wholebrain_r2_path(sub),   out_dir / 'r2_wholebrain.nii.gz')
    shutil.copy(_wholebrain_cvr2_path(sub), out_dir / 'cv_r2_wholebrain.nii.gz')
    shutil.copy(_brain_mask_path(sub), out_dir / 'brain_mask.nii.gz')

    # --- NPCr single-trial data ----------------------------------------------
    betas, voxel_xyz, masker = load_npcr_single_trials(sub)
    paradigm = load_paradigm(sub)
    if len(paradigm) != len(betas):
        raise RuntimeError(
            f'paradigm rows ({len(paradigm)}) != trials ({len(betas)})')

    betas.to_csv(npcr_dir / 'single_trial_betas.tsv.gz',
                 sep='\t', index=True, compression='gzip')
    paradigm.to_csv(npcr_dir / 'paradigm.tsv', sep='\t', index=False)

    voxel_df = pd.DataFrame(voxel_xyz, columns=['x', 'y', 'z'])
    voxel_df.index.name = 'voxel'
    voxel_df.to_csv(npcr_dir / 'voxel_coords_mm.tsv', sep='\t')

    r2_npcr = load_npcr_r2(sub, masker)
    pd.Series(r2_npcr, name='r2').to_frame().to_csv(
        npcr_dir / 'r2.tsv', sep='\t', index_label='voxel')

    cvr2_npcr = load_npcr_cvr2(sub, masker)
    pd.Series(cvr2_npcr, name='cv_r2').to_frame().to_csv(
        npcr_dir / 'cv_r2.tsv', sep='\t', index_label='voxel')

    # --- surface patch --------------------------------------------------------
    # Choose hemisphere with more voxel hits (NPCr should be right).
    best_hemi, best_hit_d = None, np.inf
    hemi_results = {}
    for hemi in ('L', 'R'):
        verts, faces = load_white_surface(sub, hemi)
        _, dist = match_voxels_to_vertices(voxel_xyz, verts)
        hemi_results[hemi] = (verts, faces, dist)
        med = float(np.median(dist))
        print(f'  hemi-{hemi}: median voxel→vertex distance {med:.2f} mm')
        if med < best_hit_d:
            best_hit_d = med
            best_hemi = hemi
    print(f'  picking hemi-{best_hemi}')
    verts, faces, _ = hemi_results[best_hemi]

    patch_verts, patch_faces, old_to_new = crop_surface_patch(
        verts, faces, voxel_xyz, PATCH_RADIUS_MM)
    voxel_to_patch_vert, voxel_to_vert_dist = match_voxels_to_vertices(
        voxel_xyz, patch_verts)
    print(f'  patch: {len(patch_verts)} vertices, {len(patch_faces)} faces')

    np.savez_compressed(
        out_dir / 'surface_patch.npz',
        vertices=patch_verts.astype(np.float32),
        faces=patch_faces.astype(np.int32),
        hemi=np.array(best_hemi),
    )
    pd.DataFrame({
        'vertex': voxel_to_patch_vert,
        'distance_mm': voxel_to_vert_dist,
    }).to_csv(npcr_dir / 'voxel_to_vertex.tsv', sep='\t', index_label='voxel')

    # --- manifest -------------------------------------------------------------
    manifest = {
        'dataset': ('Prat-Carrabin et al. 2025 — Distributed range '
                    'adaptation in human parietal encoding of numbers '
                    '(bioRxiv 10.1101/2025.09.25.675916)'),
        'subject': f'sub-{sub}',
        'model_label': MODEL,
        'smoothed': SMOOTHED,
        'hemisphere': best_hemi,
        'patch_radius_mm': PATCH_RADIUS_MM,
        'n_trials': int(len(betas)),
        'n_voxels': int(betas.shape[1]),
        'n_patch_vertices': int(len(patch_verts)),
        'n_patch_faces': int(len(patch_faces)),
        'mean_r2_npcr':    float(np.nanmean(r2_npcr)),
        'mean_cv_r2_npcr': float(np.nanmean(cvr2_npcr)),
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'  manifest: {manifest}')

    return out_dir


def zip_bundle(bundle_dir: Path, zip_path: Path) -> Path:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(bundle_dir.rglob('*')):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(bundle_dir))
    return zip_path


def main(out_zip: Path | None = None, subject: str | None = None) -> Path:
    if subject is None:
        print('Scanning subjects, scoring by NPCr tuning diversity …')
        best, best_score, info = pick_best_subject()
        print(f'\nRanking (score = √n_good · in_range_frac · √(in_range_std + 0.1)):')
        print(f'{"sub":>4}  {"score":>6}  {"n_good":>6}  '
              f'{"in_range_frac":>13}  {"in_range_std":>12}  {"med_r2":>6}')
        ranked = sorted(info.items(), key=lambda kv: -kv[1]['score'])
        for sub, d in ranked[:15]:
            marker = '  *' if sub == best else '   '
            print(f'{marker}{sub:>2}  {d["score"]:>6.2f}  {d["n_good"]:>6d}'
                  f'  {d["in_range_frac"]:>13.2f}  {d["in_range_std"]:>12.2f}'
                  f'  {d["median_r2_npcr"]:>6.3f}')
        print(f'\nBest subject: sub-{best} (score = {best_score:.2f})')
    else:
        best = subject
        print(f'Building bundle for hardcoded subject sub-{best}')

    out_zip = out_zip or Path(tempfile.gettempdir()) / 'braincoder_npc_demo.zip'
    with tempfile.TemporaryDirectory() as tmp:
        bundle_dir = Path(tmp) / 'npc_demo'
        build_bundle(best, bundle_dir)
        zip_bundle(bundle_dir, out_zip)
    size_mb = out_zip.stat().st_size / 1e6
    print(f'\nWrote {out_zip}  ({size_mb:.2f} MB)')
    return out_zip


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=None,
                   help='Output zip path (default: $TMPDIR/braincoder_npc_demo.zip)')
    p.add_argument('--subject', type=str, default=None,
                   help='Hardcode the subject id (skip ranking). E.g., --subject 38')
    args = p.parse_args()
    main(args.out, args.subject)
