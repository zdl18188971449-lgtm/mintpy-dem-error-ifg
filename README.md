# MintPy interferogram DEM-error correction

`mintpy_dem_error_ifg.py` reads a MintPy `ifgramStack.h5`, estimates one
residual DEM-error value per pixel from the selected interferogram network,
reconstructs the DEM phase for every interferogram, and writes corrected
interferogram stacks without modifying the source file.

The HDF5/block-processing workflow follows the practical pattern used by
[vceSAR](https://github.com/ymcmrs/vceSAR). The DEM inversion is implemented
independently in the interferogram domain.

## Observation model

For interferogram `master_slave` and pixel `p`:

```text
unwrapPhase = K_dem * delta_h + G_deformation * parameters + residual

K_dem = (-4*pi/wavelength) * Bperp / (slantRange * sin(incidenceAngle))
```

The constant deformation term cancels between the two acquisitions. The tool
supports polynomial, periodic, and step deformation terms.

## Classic model presets

| Model | Temporal model | Estimator |
|---|---|---|
| `linear_ols` | linear velocity | equal-weight OLS |
| `quadratic_ols` | velocity + acceleration | equal-weight OLS, MintPy default class |
| `linear_velocity` | linear | phase-velocity OLS |
| `linear_wls` / `quadratic_wls` | linear / quadratic | coherence-derived WLS |
| `linear_huber` / `quadratic_huber` | linear / quadratic | Huber IRLS |
| `linear_vce` / `quadratic_vce` | linear / quadratic | external vceSAR variance WLS |
| `adaptive_vce_huber_graph` | per-pixel linear/quadratic BIC selection | VCE + coherence weighting, Huber IRLS, observability-adaptive terrain graph |

Coherence weights use the classic relative phase-variance relation
`weight proportional to coherence^2 / (1 - coherence^2)`. VCE models accept a
dataset shaped `(num_ifg,)`, `(num_ifg, length, width)`, or vceSAR-style
`model_parameters` shaped `(num_ifg, >=3)`, where sill plus nugget is used as
the variance.

## Preflight

Run a read-only check first:

```bash
/home/zdl/miniforge3/envs/SAR/bin/python \
  /mnt/d/WSL/ALOS/Scripts/mintpy_dem_error_ifg.py \
  inputs/ifgramStack.h5 \
  -g inputs/geometryRadar.h5 \
  --models linear_ols quadratic_ols linear_wls linear_huber \
  --dry-run
```

The preflight reports network components and the representative design-matrix
rank. A disconnected network is not silently rejected because a temporal model
can make the numerical system full rank, but the resulting cross-component
calibration is model-controlled and should not be treated as an independent
observation.

## Run

```bash
/home/zdl/miniforge3/envs/SAR/bin/python \
  /mnt/d/WSL/ALOS/Scripts/mintpy_dem_error_ifg.py \
  inputs/ifgramStack.h5 \
  -g inputs/geometryRadar.h5 \
  -o dem_error_ifg_compare \
  --models linear_ols quadratic_ols linear_velocity linear_wls linear_huber \
  --mask maskTempCoh.h5 \
  --min-coherence 0.3
```

For vceSAR weighting:

```bash
... --models linear_vce quadratic_vce \
    --variance-file variogramStackModel.h5 \
    --variance-dataset model_parameters
```

To compare the classic vceSAR covariance models, first generate separate
variance-model files with `vce_modeling.py --model spherical`, `gaussian`, or
`exponential`, then run the corresponding `*_vce` preset into separate output
directories. The DEM tool deliberately consumes the variance result instead of
duplicating vceSAR's variogram fitting code.

The advanced model requires a VCE variance file and a `height` dataset in the
geometry file for terrain-guided graph edges:

```bash
python mintpy_dem_error_ifg.py inputs/ifgramStack.h5 \
  -g inputs/geometryRadar.h5 \
  --mask maskTempCoh.h5 \
  --variance-file variogramStackModel.h5 \
  --variance-dataset model_parameters \
  --models adaptive_vce_huber_graph \
  --graph-lambda 8 \
  -o dem_error_advanced
```

It first selects linear or quadratic deformation independently at every pixel,
then regularizes the DEM estimate using normalized Fisher information and
terrain/DEM-error edge weights. If geometry `height` is absent, the command
falls back to a uniform spatial graph and prints a warning.

Optional deformation terms:

```bash
... --periodic 1.0 0.5 --step-date 20080512
```

## Outputs

Each model writes:

- `ifgramStack_demErr_MODEL.h5`: MintPy-compatible stack with corrected
  `unwrapPhase` and all other source datasets copied unchanged.
- `demComponent_MODEL.h5`: `demPhase` for every interferogram and the 2D
  `demError` map in meters.
- `ifgram_metrics_MODEL.csv`: per-interferogram RMS diagnostics.
- `model_comparison.csv`: model-level DEM statistics and residual RMS.

Invalid pixels retain their original phase in the corrected stack, receive a
zero DEM phase component, and are stored as `NaN` in the DEM-error map.

`adaptive_vce_huber_graph` additionally writes `demErrorStd`, normalized
`demObservability`, and `deformationModelOrder` (`1` linear, `2` quadratic).
The current standard deviation is a local robust approximation before graph
regularization; it is not yet a fully calibrated posterior interval.

## Interpretation limits

- The method estimates relative DEM error in the spatial reference frame of
  the input interferograms.
- A connected, redundant interferogram network is required for defensible
  separation of deformation and DEM error.
- Atmospheric, ionospheric, orbit, and unwrapping residuals can leak into the
  DEM estimate. Correct these first or use robust/VCE weighting.
- A lower in-sample residual is not sufficient evidence of a better model.
  Compare held-out interferograms, baseline correlation, and external
  ICESat-2/LiDAR/GNSS elevations.
- OLS, coherence WLS, Huber, and VCE-WLS minimize different objective
  functions. Compare their DEM maps and independent validation errors; do not
  rank a weighted model only by its unweighted training RMS.

## Tests

```bash
/home/zdl/miniforge3/envs/SAR/bin/python \
  /mnt/d/WSL/ALOS/Scripts/test_mintpy_dem_error_ifg.py
```

The repository also includes a real `16 x 16` HFT-473 MintPy fixture and a
full-scene comparison for the highest-quality interferogram. See
[`TESTING.md`](TESTING.md) for reproducible commands and expected results.

## Synthetic benchmark

Generate a multi-temporal MintPy stack with known DEM-error truth, run all
classic models, compute truth-based metrics, and draw DEM/interferogram
comparison figures:

```bash
python run_synthetic_dem_benchmark.py -o synthetic_benchmark --overwrite
```

See [`SIMULATION.md`](SIMULATION.md) for the component model, controlled
research scenarios, output structure, and publication-oriented experiment
matrix.

Run the five-terrain comparison and cross-terrain model ranking with:

```bash
python run_multi_terrain_benchmark.py -o multi_terrain_benchmark --overwrite
```

## Published-method reproductions

Six literature methods are implemented in a separate numerical module so the
existing production correction path remains unchanged. Run them on a MintPy
stack with:

```bash
python run_published_models_on_mintpy.py \
  test_data/HFT473_16x16/ifgramStack.h5 \
  -g test_data/HFT473_16x16/geometryRadar.h5 \
  --mask test_data/HFT473_16x16/maskTempCoh.h5 \
  --models ica_2019 adaptive_ht_2021 pgdc_2025 \
  -o published_dem_error_ifg --overwrite
```

Run the scenario-matched comparison against the current adaptive graph model:

```bash
python run_published_model_benchmark.py \
  -o published_model_benchmark --size 18 --seed 20260713 --overwrite
```

See [`PUBLISHED_METHODS.md`](PUBLISHED_METHODS.md) for equations, reproduction
status, limitations, output files, and references. In particular, the 2015
fractal method is an adapted post-regularizer because a normal MintPy stack
does not contain the interferometric magnitude required by the paper.

The integrated `hybrid_optimal_2026` model combines the strongest applicable
parts of the published methods using per-pixel statistical gates:

```bash
python run_published_models_on_mintpy.py \
  test_data/HFT473_16x16/ifgramStack.h5 \
  -g test_data/HFT473_16x16/geometryRadar.h5 \
  --mask test_data/HFT473_16x16/maskTempCoh.h5 \
  --models hybrid_optimal_2026 \
  -o hybrid_dem_error_ifg --overwrite
```

See [`HYBRID_OPTIMAL_2026.md`](HYBRID_OPTIMAL_2026.md) for the model-selection
equations, HDF5 diagnostics, benchmark results, ablation plan, and publication
limitations.

Generate the IEEE TGRS-style spatial and performance comparison figures with:

```bash
python plot_tgrs_hybrid_comparison.py published_model_benchmark --dpi 600
```

See [`TGRS_FIGURE_NOTES.md`](TGRS_FIGURE_NOTES.md) for manuscript-ready
captions, method differences, quantitative interpretation, and validation
limits.

## Statistical and real-data validation

Run 30 paired simulations with 95% confidence intervals and module ablations:

```bash
python run_homa_statistical_validation.py \
  -o homa_statistical_validation --replicates 30 \
  --seed-start 20260713 --size 18 --workers 4 --dpi 600 --overwrite
```

Run connected-network interferogram holdout validation and, when an independent
georeferenced DEM/DSM is available, external elevation validation:

```bash
python validate_homa_real_data.py \
  test_data/HFT473_16x16/ifgramStack.h5 \
  -g test_data/HFT473_16x16/geometryRadar.h5 \
  --mask test_data/HFT473_16x16/maskTempCoh.h5 \
  --lookup /path/to/geometryGeo.h5 \
  --external-dem /path/to/external_dem.tif \
  -o homa_real_validation --folds 5 --dpi 600 --overwrite
```

See [`VALIDATION.md`](VALIDATION.md) for the exact evaluation protocol, output
tables, Copernicus GLO-30 example, negative results, and publication limits.

## HFT-47 full-scene processing

Apply the validation-driven static HOMA profile to a complete MintPy stack with
overlapping row strips and a residual-safe global DEM scale:

```bash
python run_homa_full_scene.py \
  /path/to/inputs/ifgramStack.h5 \
  -g /path/to/inputs/geometryRadar.h5 \
  --mask /path/to/maskTempCoh.h5 \
  --mintpy-dem /path/to/demErr.h5 \
  --timeseries-before /path/to/timeseries_tropHgt.h5 \
  --timeseries-mintpy-corrected /path/to/timeseries_tropHgt_demErr.h5 \
  -o /path/to/homa_dem_correction \
  --block-rows 32 --halo-rows 8 --workers 8 --dpi 600 --overwrite
```

See [`HFT47_FULL_SCENE_RESULTS.md`](HFT47_FULL_SCENE_RESULTS.md) for the HFT-473
and HFT-474 commands, output products, comparison figures, quantitative results,
and the HFT-474 limited-observability warning.
