# Synthetic interferogram DEM-error benchmark

This workflow adapts the component-based strategy in
`InterferogramSimulator-main/README.md` to the multi-temporal observation
model required by MintPy. The original simulator produces independent wrapped
samples for denoising, deformation detection, and phase unwrapping. DEM-error
inversion instead requires multiple acquisitions, a connected interferogram
network, perpendicular baselines, geometry, and a common DEM-error truth map.

## Simulation model

For acquisition dates `a` and `b`, the simulated unwrapped phase is:

```text
phase_ab = K_ab * dem_error
         + deformation_b - deformation_a
         + atmosphere_b - atmosphere_a
         + interferogram_turbulence
         + coherence_dependent_noise
         + optional_unwrapping_cycle
```

The generator creates these components independently:

- terrain: smooth random relief, a Gaussian hill, and a regional slope;
- DEM error: terrain-correlated error, a smooth regional error, a localized
  Gaussian error, and a sharp block boundary;
- deformation: spatially varying velocity and acceleration;
- atmosphere: acquisition-level turbulent and stratified fields plus a small
  interferogram-specific turbulent field;
- decorrelation: terrain/deformation-dependent coherence and a water mask;
- unwrapping error: a known integer-cycle error in one interferogram;
- geometry: spatially varying incidence angle, slant range, and acquisition
  perpendicular-baseline grids.

Acquisition-level atmosphere and deformation are differenced when forming an
interferogram. This preserves network closure more realistically than drawing
every interferogram independently.

## Run

Install the same lightweight dependencies used by the correction project:

```bash
python -m pip install numpy h5py matplotlib
```

Run the default 12-acquisition benchmark:

```bash
python run_synthetic_dem_benchmark.py \
  -o synthetic_benchmark \
  --overwrite
```

Select a terrain-specific DEM-error model with `--terrain-type`. Available
types are `plain`, `hills`, `mountain`, `valley`, and `urban`.

## Multi-terrain benchmark

Run all five terrain types with the same acquisition-network settings, all
eight classic correction models, and the adaptive VCE-Huber graph model:

```bash
python run_multi_terrain_benchmark.py \
  -o multi_terrain_benchmark \
  --overwrite
```

Use repeated random realizations for statistical experiments:

```bash
python run_multi_terrain_benchmark.py \
  -o multi_terrain_replicates \
  --replicates 10 \
  --overwrite
```

The terrain cases represent:

- `plain`: low relief, regional tilt, field-block errors, and a meandering river;
- `hills`: smooth hills, terrain-correlated error, and localized DEM bias;
- `mountain`: steep ridges, peaks, slope-dependent error, and low coherence;
- `valley`: an anisotropic valley/river axis and opposite side-slope biases;
- `urban`: low regional relief with sharp building-height and footprint errors.

Within each replicate, all terrain types share the same acquisition dates,
baseline realization, atmospheric/noise random stream, and unwrapping-error
setting. Terrain generation uses an independent random stream, so differences
between terrain columns are not caused by changing the network or noise seed.

The batch command adds `multi_terrain_metrics.csv`, an across-replicate
`multi_terrain_summary.csv`, an RMSE heatmap/ranking figure, and a terrain,
truth, and best-model error gallery. Each terrain/replicate subdirectory also
contains the complete single-scene MintPy products and figures.

The command generates the MintPy inputs, runs all linear/quadratic OLS,
coherence-WLS, Huber, VCE-WLS, and the adaptive graph preset, scores them
against truth, and draws the comparison figures.

For a quick smoke test:

```bash
python run_synthetic_dem_benchmark.py \
  -o synthetic_smoke \
  --size 32 \
  --num-acquisitions 8 \
  --max-neighbor 2 \
  --models linear_ols quadratic_ols linear_huber quadratic_huber \
  --overwrite
```

## Research scenarios

Use the following controlled experiments. Keep the seed list identical across
models and report the distribution over at least 100 simulations, not only one
favorable realization.

```bash
# Ideal model-identification case
python run_synthetic_dem_benchmark.py -o runs/ideal \
  --atmosphere-scale 0 --noise-scale 0 --outlier-cycles 0 --overwrite

# Strong spatially correlated atmosphere
python run_synthetic_dem_benchmark.py -o runs/atmosphere \
  --atmosphere-scale 2 --outlier-cycles 0 --overwrite

# One-cycle unwrapping error: tests Huber robustness
python run_synthetic_dem_benchmark.py -o runs/outlier \
  --outlier-cycles 1 --overwrite

# DEM/deformation identifiability stress test
python run_synthetic_dem_benchmark.py -o runs/correlation \
  --baseline-time-correlation 0.85 --overwrite

# Linear deformation truth: quadratic models should not gain a truth advantage
python run_synthetic_dem_benchmark.py -o runs/linear_truth \
  --acceleration-scale 0 --overwrite
```

Recommended factorial levels are baseline-time correlation
`[0, 0.5, 0.85]`, atmosphere scale `[0, 1, 2]`, outlier cycles `[0, 1]`, and
acceleration scale `[0, 1]`.

## Outputs

- `inputs/ifgramStack.h5`: simulated MintPy interferogram stack;
- `inputs/geometryRadar.h5`: geometry and pixel-wise acquisition baselines;
- `maskTempCoh.h5`: valid non-water pixels;
- `simulationVariance.h5`: known per-interferogram noise variance for VCE-WLS;
- `simulationTruth.h5`: DEM error and every simulated phase component;
- `correction/`: normal outputs from `mintpy_dem_error_ifg.py`;
- `benchmark_metrics.csv`: truth-based DEM and phase metrics;
- `synthetic_dem_comparison.png/.pdf`: truth, best estimates, errors, and all-model RMSE;
- `synthetic_ifgram_comparison.png/.pdf`: observed phase, true/estimated DEM
  phase, corrected phase, and correction error.

Model selection should prioritize `dem_rmse_m`, `dem_phase_rmse_rad`, and
`corrected_truth_rmse_rad`. For cross-terrain comparison, also use `dem_nrmse`,
defined as DEM RMSE divided by the standard deviation of the true DEM-error
field. The in-sample network residual is retained for
diagnosis but is not a ground-truth accuracy measure.

## Current advanced-model result

For the bundled five-terrain experiment with 12 acquisitions, 30
interferograms, one-cycle unwrapping contamination, and one realization per
terrain, the cross-terrain mean DEM RMSE is:

| Model | Mean DEM RMSE |
|---|---:|
| `adaptive_vce_huber_graph` | 2.491 m |
| `linear_huber` | 2.603 m |
| `quadratic_huber` | 2.611 m |

The advanced model improves over `linear_huber` by about 4.3% and ranks first
for every terrain type. With unwrapping contamination disabled, the means are
2.506 m and 2.622 m, respectively, so the graph/adaptive improvement is not
only an outlier effect.

These numbers are implementation checks, not publication evidence. The
8-acquisition smoke test is less stable and can favor `linear_huber`; use a
connected, redundant network and select graph/BIC parameters with held-out
interferograms rather than truth from the evaluation set. At least 30 random
realizations and independent real sites are required for defensible claims.
