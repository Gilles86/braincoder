from importlib.resources import files
import numpy as np
import pandas as pd
from scipy import ndimage
from nilearn.surface import load_surf_data

def load_szinte2024(resize_factor=1., best_voxels=None):
    data = {}

    base = files(__package__) / '../data/szinte2024'

    with (base / 'dm_gazecenter.npz').open('rb') as f:
        data['stimulus'] = np.load(f)['arr_0'].astype(np.float32)

    with (base / 'grid_coordinates.tsv').open('rb') as f:
        data['grid_coordinates'] = pd.read_csv(f, sep='\t').astype(np.float32)

    with (base / 'gauss_parameters.tsv').open('rb') as f:
        prf_pars = pd.read_csv(f, index_col='source', sep='\t').astype(np.float32)

    if resize_factor != 1.:
        data['stimulus'] = np.array([ndimage.zoom(d, 1./resize_factor) for d in data['stimulus']]).astype(np.float32)

        tmp_x = data['grid_coordinates'].set_index(pd.MultiIndex.from_frame(data['grid_coordinates']))['x']
        tmp_y = data['grid_coordinates'].set_index(pd.MultiIndex.from_frame(data['grid_coordinates']))['y']

        new_x = ndimage.zoom(tmp_x.unstack('y'), 1./resize_factor)
        new_y = ndimage.zoom(tmp_y.unstack('y'), 1./resize_factor)

        data['grid_coordinates'] = pd.DataFrame({'x': new_x.ravel(), 'y': new_y.ravel()[::-1]}).astype(np.float32)

    with (base / 'mask-v1_bold.tsv.gz').open('rb') as f:
        data['v1_timeseries'] = pd.read_csv(f, sep='\t', index_col='time', compression='gzip').astype(np.float32)

    data['v1_timeseries'].columns = data['v1_timeseries'].columns.astype(int)
    data['tr'] = 1.317025

    if best_voxels:
        best_voxels = prf_pars['r2'].sort_values(ascending=False).iloc[:best_voxels].index
        data['v1_timeseries'] = data['v1_timeseries'].loc[:, best_voxels]
        prf_pars = prf_pars.loc[best_voxels]

    data['prf_pars'] = prf_pars
    data['r2'] = prf_pars['r2']
    data['prf_pars'].drop('r2', axis=1, inplace=True)

    return data

import os
import pathlib
import requests
import zipfile
import shutil
from tqdm import tqdm

FIGSHARE_URL = "https://figshare.com/ndownloader/files/26577941"
DATA_DIR = pathlib.Path.home() / ".braincoder" / "data"
ZIP_PATH = DATA_DIR / "fmri_teaching_materials.zip"

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    directory.mkdir(parents=True, exist_ok=True)

def download_file(url, save_path):
    """Download a file from a URL with a progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise error if download fails

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading")

    with open(save_path, "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)

    progress_bar.close()

def extract_zip(zip_path, extract_to):
    """Extracts the given zip file."""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

def load_vanes2019(raw_files=False, downsample_stimulus=5.):
    """
    Ensures the fMRI van Es 2019 dataset is available in ~/.braincoder/data.
    Downloads and extracts it if necessary.
    Returns the path to the dataset.
    """
    ensure_directory_exists(DATA_DIR)
    dataset_folder = DATA_DIR / "prf_vanes2019"

    if not dataset_folder.exists() or not any(dataset_folder.iterdir()):
        print(f"Dataset missing or incomplete in {dataset_folder}, redownloading...")
        if dataset_folder.exists():
            shutil.rmtree(dataset_folder)  # Remove partial data
        download_file(FIGSHARE_URL, ZIP_PATH)
        print("\nDownload complete, extracting...")
        extract_zip(ZIP_PATH, dataset_folder)
        print("\nExtraction complete, cleaning up...")
        ZIP_PATH.unlink()  # Remove zip file after extraction

    data_lh = load_surf_data(dataset_folder / "sub-02_task-prf_space-59k_hemi-L_run-median_desc-bold.func.gii")
    data_rh = load_surf_data(dataset_folder / "sub-02_task-prf_space-59k_hemi-R_run-median_desc-bold.func.gii")

    if raw_files:
        # Return a list with all files in dataset_folder:
        return list(dataset_folder.iterdir())

    data = {}
    data['ts'] = pd.concat((pd.DataFrame(data_lh, index=pd.Index(np.arange(len(data_lh)), name='vertex')),
                      pd.DataFrame(data_rh, index=pd.Index(np.arange(len(data_rh)), name='vertex'))), keys=['L', 'R'], names=['hemisphere'], axis=0).T

    # Convert ts to percent signal change:
    data['ts'] = (data['ts'] - data['ts'].mean()) / data['ts'].mean() * 100

    data['stimulus'] = zoom(io.loadmat(dataset_folder / "vis_design.mat")['stim'], (1./downsample_stimulus, 1./downsample_stimulus, 1)).astype(np.float32)
    data['stimulus'] = np.clip(np.moveaxis(np.moveaxis(data['stimulus'], -1, 0), -1, 1) / 255., 0, 1)

    # Calculate the degree per pixel scaling factors
    width_pixels, height_pixels = data['stimulus'].shape[1:]
    width_degrees = 20
    dx = width_degrees / width_pixels
    dy = dx
    height_degrees = height_pixels * dy

    x_degrees = np.linspace(-width_degrees / 2 + dx / 2, width_degrees / 2 - dx / 2, width_pixels)
    y_degrees = np.linspace(-height_degrees / 2 + dy / 2, height_degrees / 2 - dy / 2, height_pixels)

    x_mesh, y_mesh = np.meshgrid(x_degrees, y_degrees)

    data['grid_coordinates'] = pd.DataFrame({'x': x_mesh.ravel(), 'y': y_mesh.ravel()}).astype(np.float32)
    data['tr'] = 1.5

    return data


# Numerosity demo bundle (sub-13 from the Prat-Carrabin et al. 2025
# numerosity fMRI dataset). The release URL is overridable for local
# testing via the env var BRAINCODER_NPC_DEMO_URL — point that at a
# file:// or http(s):// resource.
# Article: https://doi.org/10.6084/m9.figshare.32358570 (Prat-Carrabin et al.
# 2025 numerosity NPC extract, sub-13).
NPC_DEMO_DEFAULT_URL = "https://ndownloader.figshare.com/files/64795824"


def load_pratcarrabin2025_npc(force_redownload=False):
    """Download and load the numerosity demo bundle.

    A small (~4 MB) extract from the numerosity fMRI dataset accompanying
    Prat-Carrabin et al. (2025, *Distributed range adaptation in human
    parietal encoding of numbers*, bioRxiv 10.1101/2025.09.25.675916),
    built from one subject (highest mean NPCr CV-R² with model 15). The
    bundle is structured so the encoding-pipeline example notebooks
    (``examples/01_decoding_pipeline``) can run end-to-end without any
    local BIDS dataset on disk.

    Returns
    -------
    dict with keys::

        manifest                 — provenance dict (subject id, hemisphere, ...)
        r2_wholebrain            — nibabel Nifti1Image, whole-brain
                                   within-sample R² (use for the mixture-
                                   model voxel selection)
        cv_r2_wholebrain         — nibabel Nifti1Image, whole-brain CV-R²
                                   (for generalisation reporting)
        brain_mask               — nibabel Nifti1Image, EPI brain mask (T1w)
        data                     — pd.DataFrame (n_trials × n_vox), NPCr betas
        paradigm                 — pd.DataFrame (n_trials × {session, run,
                                   trial_nr, n, range}); ``n`` is the
                                   presented numerosity, ``range`` ∈
                                   {'narrow', 'wide'}
        voxel_coords_mm          — pd.DataFrame (n_vox × {x, y, z}), T1w mm
        voxel_to_vertex          — pd.DataFrame (n_vox × {vertex,
                                   distance_mm}) — nearest patch vertex per
                                   voxel
        r2                       — pd.Series, NPCr within-sample R² per voxel
        cv_r2                    — pd.Series, NPCr CV-R² per voxel
        surface_vertices         — (n_patch_vertices × 3) float32 array
        surface_faces            — (n_patch_faces × 3) int32 array
        hemisphere               — 'L' or 'R'
    """
    url = os.environ.get('BRAINCODER_NPC_DEMO_URL', NPC_DEMO_DEFAULT_URL)
    dataset_folder = DATA_DIR / 'pratcarrabin2025_npc'
    zip_path = DATA_DIR / 'pratcarrabin2025_npc.zip'

    ensure_directory_exists(DATA_DIR)
    if force_redownload and dataset_folder.exists():
        shutil.rmtree(dataset_folder)

    if not dataset_folder.exists() or not any(dataset_folder.iterdir()):
        if url.startswith('file://') or os.path.isabs(url):
            # Local-file shortcut for development.
            src = url.removeprefix('file://')
            print(f'Copying demo bundle from {src} …')
            shutil.copy(src, zip_path)
        else:
            print(f'Downloading demo bundle from {url} …')
            download_file(url, zip_path)
        ensure_directory_exists(dataset_folder)
        extract_zip(zip_path, dataset_folder)
        zip_path.unlink()

    import json
    import nibabel as nib

    manifest = json.loads((dataset_folder / 'manifest.json').read_text())

    r2_wholebrain    = nib.load(dataset_folder / 'r2_wholebrain.nii.gz')
    cv_r2_wholebrain = nib.load(dataset_folder / 'cv_r2_wholebrain.nii.gz')
    brain_mask       = nib.load(dataset_folder / 'brain_mask.nii.gz')

    npcr = dataset_folder / 'npcr'
    data = pd.read_csv(npcr / 'single_trial_betas.tsv.gz',
                       sep='\t', index_col='trial', compression='gzip')
    data.columns = data.columns.astype(int)
    data.columns.name = 'voxel'
    data = data.astype(np.float32)

    paradigm = pd.read_csv(npcr / 'paradigm.tsv', sep='\t')

    voxel_coords_mm = pd.read_csv(npcr / 'voxel_coords_mm.tsv',
                                  sep='\t', index_col='voxel')
    voxel_to_vertex = pd.read_csv(npcr / 'voxel_to_vertex.tsv',
                                  sep='\t', index_col='voxel')
    r2    = pd.read_csv(npcr / 'r2.tsv',    sep='\t',
                        index_col='voxel')['r2']
    cv_r2 = pd.read_csv(npcr / 'cv_r2.tsv', sep='\t',
                        index_col='voxel')['cv_r2']

    with np.load(dataset_folder / 'surface_patch.npz', allow_pickle=False) as npz:
        surface_vertices = npz['vertices'].astype(np.float32)
        surface_faces    = npz['faces'].astype(np.int32)
        hemi             = str(npz['hemi'])

    return {
        'manifest':         manifest,
        'r2_wholebrain':    r2_wholebrain,
        'cv_r2_wholebrain': cv_r2_wholebrain,
        'brain_mask':       brain_mask,
        'data':             data,
        'paradigm':         paradigm,
        'voxel_coords_mm':  voxel_coords_mm,
        'voxel_to_vertex':  voxel_to_vertex,
        'r2':               r2,
        'cv_r2':            cv_r2,
        'surface_vertices': surface_vertices,
        'surface_faces':    surface_faces,
        'hemisphere':       hemi,
    }
