# Can a linear map predict the changes lost by averaging?

September 13, 2026. This is an exploratory activation diagnostic, motivated
by the selected mean direction tying the `First,` baseline in the primary
test. It does not change that experiment or establish a new accuracy result.

Use the 128 saved extraction questions and predict the difference between
prefix-conditioned and zero-shot activations from the zero-shot activation.
Eight shuffled folds with seed 907 give every question a prediction from a
fit that excludes it. Compare the training-fold mean, a scalar rescaling of
the centered zero-shot activation plus the mean, and a ridge linear map.
The ridge penalty is fixed at the mean diagonal of the training Gram matrix.
There is no hyperparameter search and no mathematical answer enters the fit.

The table reports the reduction in prediction error relative to the
training-fold mean, evaluated outside the training folds. It measures the
remaining error, not the total activation energy or math accuracy.

| Block index | Scalar rescaling, real ICL | Linear map, real ICL | Linear map, rotated pairs | Linear map, shuffled tokens | Linear map, filler |
|---|---:|---:|---:|---:|---:|
| 7 | 34.4% | 51.2% | 51.6% | 48.5% | 46.5% |
| 13 | 43.0% | 45.0% | 48.0% | 35.4% | 34.1% |
| 20 | 48.1% | 36.8% | 39.2% | 26.3% | 30.3% |
| 27 | 42.9% | 33.8% | 42.0% | 24.6% | 21.6% |

The mean alone accounts for 87.9% of real ICL difference energy at block 7
under this cross-validation procedure. The linear map removes about half
the remaining error. Its improvement occurs in every fold, ranging from
43.4% to 56.1%. Thus, averaging does discard predictable changes that depend
on the question.

However, this predictability is not specific to correct demonstrations.
Rotated and shuffled prefixes show similar results, and simple scalar
rescaling beats the ridge map at the deeper blocks. A future causal test
would need to compare any conditional intervention with these controls on
fresh questions. Better activation reconstruction alone is insufficient.
The overlapping training folds and single demonstration bank also limit
the interpretation. Full results, fold assignments, and source hashes are
in [the geometry artifact](../results/gsm8k-v2-conditional-geometry.json).

## Numerical verification

Sanity tests exposed incorrect Torch 2.7.0 CPU float64 matrix operations on
this node. On a well-conditioned synthetic system, its solve had a residual
norm of about 227,586, while NumPy's residual was below 2e-14. A real
activation Gram matrix computed by Torch was not symmetric. The cause in
the numerical software stack has not been established.

The diagnostic therefore uses NumPy for matrix products and solves, with a
residual check on every solve. Four tests cover exclusion of held-out
targets, recovery of a known linear map, a map with 3,584 features, and
exact scalar rescaling. All pass.

An independent NumPy audit agrees with the saved centered-SVD statistics
within 1.2e-7 and with projection energies within 3.8e-7. The original mean
direction, GPU generations, and answer scoring do not use the failed CPU
solver. No shared environment settings were changed. The numerical evidence
and reproducible matrix definitions are in
[the CPU audit](../results/gsm8k-v2-cpu-numerics.json).

Run the diagnostic on the server with two CPU threads:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src \
  .venv-benchmark/bin/python -m analysis.gsm8k_conditional_geometry \
  --run runs/gsm8k-steering-v2 --controls runs/gsm8k-prefix-controls-v1 \
  --output runs/gsm8k-conditional-geometry-v1/metrics.json
```
