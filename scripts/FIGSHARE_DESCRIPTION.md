# Braincoder NPC demo bundle: single-trial numerosity fMRI extract (Prat-Carrabin et al. 2025, sub-13)

## Description

A small (~4 MB) derived data bundle used by the [`braincoder`](https://github.com/Gilles86/braincoder) Python package's tutorial notebooks (`examples/01_decoding_pipeline/`). It contains one subject's right numerosity-tuned parietal cortex (NPCr) extract from the numerosity fMRI experiment described in Prat-Carrabin et al. (2025; see below), repackaged for fast, one-line download via `braincoder.utils.data.load_pratcarrabin2025_npc()`.

The bundle was built by selecting the subject with the highest mean NPCr cross-validated R² under a LogGaussian-PRF numerosity model (sub-13), then extracting only the artefacts the tutorial needs.

This bundle is intended **only as a teaching / demonstration resource**. For any actual analysis, please work from the full BIDS dataset.

## Contents

| File | Description |
|---|---|
| `r2_wholebrain.nii.gz` | Whole-brain within-sample R² (T1w space) — used for the mixture-model voxel-selection example. |
| `cv_r2_wholebrain.nii.gz` | Whole-brain cross-validated R² (T1w space) — for generalisation reporting. |
| `brain_mask.nii.gz` | fmriprep EPI brain mask (T1w space). |
| `npcr/single_trial_betas.tsv.gz` | Single-trial GLM beta estimates (480 trials × 852 voxels) inside NPCr. |
| `npcr/paradigm.tsv` | Per-trial stimulus: numerosity 10–40, narrow / wide range conditions. |
| `npcr/voxel_coords_mm.tsv` | NPCr voxel centroid coordinates (T1w mm). |
| `npcr/voxel_to_vertex.tsv` | Nearest cortical-surface vertex per NPCr voxel (for geodesic distance). |
| `npcr/r2.tsv` | Per-voxel within-sample R² inside NPCr. |
| `npcr/cv_r2.tsv` | Per-voxel cross-validated R² inside NPCr. |
| `surface_patch.npz` | 30 mm-radius patch of the right hemisphere white-matter mesh (vertices + faces) cropped around NPCr. |
| `manifest.json` | Provenance metadata (subject id, hemisphere, patch radius, voxel / vertex counts, mean R²). |

## Source dataset & citation

This bundle is derived from the numerosity fMRI dataset accompanying:

> Prat-Carrabin, A., de Hollander, G., Bedi, S., Gershman, S. J., & Ruff, C. C. (2025). *Distributed range adaptation in human parietal encoding of numbers.* bioRxiv. <https://doi.org/10.1101/2025.09.25.675916>

<!-- TODO: once the full BIDS dataset is published on OpenNeuro, add the accession + DOI here, e.g.: -->
<!-- OpenNeuro accession: dsXXXXXX, https://openneuro.org/datasets/dsXXXXXX -->

Please cite the paper above as the primary data source in any work derived from this bundle.

## Software

`braincoder` — <https://github.com/Gilles86/braincoder>

## License

CC0 1.0 Universal — to remain compatible with the upstream BIDS dataset release (ODC Public Domain Dedication and License v1.0).

## Suggested citation for this bundle

> de Hollander, G. (2026). *Braincoder NPC demo bundle: single-trial numerosity fMRI extract (Prat-Carrabin et al. 2025, sub-13).* Figshare. [DOI assigned on upload]

## Keywords

fMRI · numerosity · population receptive fields · encoding models · Bayesian decoding · braincoder · sample data · tutorial · parietal cortex · cortical geodesic distance
