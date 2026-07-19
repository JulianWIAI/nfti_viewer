"""
dti/ — Diffusion Tensor Imaging tractography package
═══════════════════════════════════════════════════════
Sub-modules (each in its own file per the modularisation policy):

  config.py      TractographyConfig dataclass — all tuning parameters
  gradients.py   Parse .bval / .bvec bytes → dipy GradientTable
  masking.py     Brain extraction via median_otsu (skull strip)
  tensor_fit.py  Tensor model fitting, FA / MD / eigenvector maps
  tracking.py    Deterministic LocalTracking pipeline
  compression.py Streamline length-filter, subsampling, Douglas–Peucker decimation
                 + JSON serialisation

Typical pipeline order:
  1. build_gradient_table()   → GradientTable
  2. extract_brain_mask()     → (masked_data, mask)
  3. fit_tensor_model()       → TensorFitResult
  4. run_tractography()       → raw Streamlines
  5. compress_and_filter()    → decimated Streamlines
  6. serialize_streamlines()  → list[list[list[float]]] (JSON-ready)
"""
