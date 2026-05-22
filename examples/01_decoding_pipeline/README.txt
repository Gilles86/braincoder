######################################
End-to-end decoding pipeline on real data
######################################

Three connected examples that walk through a full encoding-model decoding
pipeline on a small extract of single-trial fMRI data (numerosity task,
right numerosity-tuned parietal cortex):

1. Select responsive voxels from whole-brain R² by fitting a 2-component
   mixture and picking a threshold (FDR or posterior).
2. Decode trial-wise stimulus posteriors using a noise model whose
   covariance is regularised by *geodesic* distance on the cortical surface.
3. Quantify *expected* decoding uncertainty by simulating from the fitted
   model and re-decoding.

Each example is self-contained but share the same demo dataset, which is
downloaded on first run via ``braincoder.utils.data.load_dehollander2024_npc``.
