# Working on braincoder itself

- Use `keras.ops`, not `tf.*`, and don't wrap what `keras.ops` already does. Test on all
  three backends (`KERAS_BACKEND=… pytest`; torch on a Mac also needs
  `PYTORCH_ENABLE_MPS_FALLBACK=1`).
- Never call `float()` or `convert_to_numpy` inside a loss, and never wrap an optimisation
  step in a bare `except Exception`. Together these caused the unfitted-Ω bug on JAX
  ([version_history.md](version_history.md)).
- When adding an `__init__` argument, check it reaches the optimisation loop:
  `ResidualFitter.lambd` was silently ignored for a long time.
- Fitting runs eagerly on all backends in the Keras port (no `tf.function` or `jit`). Docs
  that mention `@tf.function` describe the pre-port code.
- After a behaviour change, run `scripts/check_skill_claims.py` (in this skill) and update
  SKILL.md where it reports CHANGED.
