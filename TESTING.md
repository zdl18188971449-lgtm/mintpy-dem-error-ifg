# Testing guide

This repository contains two levels of test data:

1. `test_data/HFT473_16x16/`: a small real MintPy fixture for fast end-to-end
   execution.
2. `examples/HFT473_20090104_20090219/`: the full-scene, `4 x 4` multilooked
   model comparison for the highest-quality interferogram in the current HFT
   dataset.

## Environment

The correction tool requires Python 3.10+ with:

```bash
python -m pip install numpy h5py matplotlib
```

MintPy is optional for the correction code itself, but recommended for opening
and continuing to process the corrected `ifgramStack` products.

## 1. Unit tests

Run the synthetic numerical and HDF5 compatibility tests:

```bash
python test_mintpy_dem_error_ifg.py -v
```

The tests verify:

- exact recovery of a known DEM-error field in a noise-free system;
- improved robustness of Huber IRLS when one interferogram contains an outlier;
- corrected phase and DEM-component HDF5 datasets;
- optional loading of the corrected stack with MintPy `ifgramStack`.

## 2. Real 16 x 16 MintPy fixture

The fixture is extracted from HFT-473 at radar-coordinate box:

```text
x: 944:960
y: 32:48
```

It contains 26 interferograms, 11 acquisitions, a connected network, and 256
valid temporal-coherence mask pixels.

Run all four comparison models:

```bash
python mintpy_dem_error_ifg.py \
  test_data/HFT473_16x16/ifgramStack.h5 \
  -g test_data/HFT473_16x16/geometryRadar.h5 \
  --mask test_data/HFT473_16x16/maskTempCoh.h5 \
  -o test_output/HFT473_16x16 \
  --models linear_ols quadratic_ols linear_wls linear_huber \
  --block-rows 8
```

Expected key console checks:

```text
stack shape: (26, 16, 16)
fit interferograms: 26 / 26
acquisitions: 11
network components: 1 [11]
valid pixels: 256 for every model
```

Compare the generated summary with the bundled reference:

```bash
cat test_output/HFT473_16x16/model_comparison.csv
cat test_data/HFT473_16x16/expected/model_comparison.csv
```

Small floating-point differences are acceptable across NumPy/LAPACK versions.
The network size, valid-pixel count, output shapes, and model ordering should
remain unchanged.

## 3. Full-scene highest-quality interferogram test

The bundled example was selected from both HFT-473 and HFT-474 using the
largest mean spatial coherence:

```text
project: HFT-473
date pair: 20090104_20090219
mean coherence: 0.9790
temporal baseline: 46 days
perpendicular baseline: 433.1 m
processing: 4 x 4 coherence-weighted looks
valid pixels: 382,844
```

To reproduce it from a MintPy work directory containing `inputs/`,
`maskTempCoh.h5`, and `coherenceSpatialAvg.txt`:

```bash
python plot_best_ifg_dem_test.py /path/to/HFT-473/Mintpy/mintpy \
  -o best_ifg_dem_test \
  --looks 4 \
  --block-rows 16 \
  --models linear_ols quadratic_ols linear_wls linear_huber
```

Bundled outputs:

- `20090104_20090219_dem_correction_comparison.png`: multi-panel comparison.
- `20090104_20090219_dem_correction_comparison.pdf`: vector figure.
- `20090104_20090219_dem_model_comparison.h5`: original phase, coherence,
  mask, corrected phase, DEM phase, and DEM-error map for every model.
- `20090104_20090219_model_metrics.csv`: model metrics.

The selected-interferogram RMS values are:

| Model | Original RMS (rad) | Corrected RMS (rad) | All-IFG residual RMS (rad) |
|---|---:|---:|---:|
| `linear_ols` | 0.2451 | 0.2622 | 0.8582 |
| `quadratic_ols` | 0.2451 | 0.2638 | 0.8489 |
| `linear_wls` | 0.2451 | 0.1998 | 1.1216 |
| `linear_huber` | 0.2451 | 0.2477 | 0.8870 |

Coherence WLS minimizes a weighted objective and performs best on the selected
high-coherence interferogram, while quadratic OLS has the lowest unweighted
residual over the complete interferogram network. Do not choose a production
model from one in-sample metric alone; use held-out interferograms and external
elevation validation where possible.

## Output HDF5 structure

Inspect the bundled comparison file:

```python
import h5py

path = "examples/HFT473_20090104_20090219/20090104_20090219_dem_model_comparison.h5"
with h5py.File(path, "r") as f:
    print(dict(f.attrs))
    print(list(f.keys()))
    print(f["linear_wls/correctedPhase"].shape)
    print(f["linear_wls/demPhase"].shape)
    print(f["linear_wls/demError"].shape)
```

All example raster datasets have shape `(759, 621)`.
