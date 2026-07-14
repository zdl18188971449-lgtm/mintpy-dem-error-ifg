# HFT-47 MintPy-Compatible Full-Scene Huber Results

## Processing contract

Both HFT-473 and HFT-474 were rerun from the exact time-series file selected by
MintPy's `correct_topography` step: `timeseries_tropHgt.h5`. The same MintPy
1.6.2 template, quadratic deformation model, acquisition dates, reference
metadata, temporal-coherence mask, incidence angle, slant range, perpendicular
baseline, corrected time-series formula, and velocity inversion were used.

The only model change was the numerical objective in the DEM-error inversion:
MintPy OLS versus Huber IRLS with delta 1.345 and eight iterations. A batched
implementation was used for speed. Against MintPy's scalar solver, the maximum
full-scene differences of the batched OLS control were `3.8e-5 m` in DEM error
and `2.1e-7 m` in corrected displacement.

## Runtime and products

| Project | Shape | Dates | Valid pixels | Huber runtime | Peak memory |
|---|---:|---:|---:|---:|---:|
| HFT-473 | 3036 x 2484 | 11 | 6,168,960 | 70.7 s | 2.7 GB |
| HFT-474 | 3092 x 2476 | 10 | 4,885,802 | 54.8 s | 2.2 GB |

Large HDF5 products are stored locally in each MintPy project under
`homa_huber_mintpy_compatible/`:

- `demErr.h5`
- `timeseries_tropHgt_demErr.h5`
- `timeseriesResidual.h5`
- `velocity.h5`

A fresh OLS control from the same current input is stored under
`mintpy_ols_current_input/`. This was necessary because the historical root
`timeseriesResidual.h5` had been regenerated from a later linear velocity fit
and was not the quadratic residual produced by the current DEM-error system.

## Full-scene metrics

| Metric | HFT-473 | HFT-474 |
|---|---:|---:|
| DEM difference mean, Huber - OLS | 0.079 m | 2.560 m |
| DEM centered difference RMSE | 0.598 m | 3.133 m |
| Residual L2 RMS, OLS | 7.156 mm | 10.645 mm |
| Residual L2 RMS, Huber | 7.385 mm | 12.226 mm |
| Median absolute residual, OLS | 3.928 mm | 5.158 mm |
| Median absolute residual, Huber | 3.220 mm | 3.493 mm |
| 95th absolute residual, OLS | 14.643 mm | 24.485 mm |
| 95th absolute residual, Huber | 15.784 mm | 26.084 mm |
| Velocity centered difference RMSE | 0.707 mm/yr | 0.017 mm/yr |

Huber lowers the median absolute residual by about 18% for HFT-473 and 32% for
HFT-474, while increasing the L2 RMS and the 95th-percentile residual. This is
expected: OLS minimizes the L2 objective, whereas Huber sacrifices the fit to a
small number of acquisitions to improve the typical residual.

For HFT-474, Huber reduces the residual RMS for the first nine acquisitions but
raises the last acquisition (`20090908`) from 20.47 to 33.59 mm. The last date
is treated as an outlier or temporal-model mismatch. The DEM changes strongly
because HFT-474 has weak DEM sensitivity, but the velocity centered RMSE changes
by only 0.017 mm/yr. This is not evidence that the Huber DEM is more accurate;
external elevation or held-out acquisition validation is still required.

## Reproduction

```bash
python mintpy_dem_error_timeseries.py \
  timeseries_tropHgt.h5 -g inputs/geometryRadar.h5 \
  -t smallbaselineApp.cfg --estimator homa_huber \
  --pixel-batch-size 32768 \
  -o homa_huber_mintpy_compatible/timeseries_tropHgt_demErr.h5 \
  --dem-err-file homa_huber_mintpy_compatible/demErr.h5

timeseries2velocity.py \
  homa_huber_mintpy_compatible/timeseries_tropHgt_demErr.h5 \
  -t smallbaselineApp.cfg \
  -o homa_huber_mintpy_compatible/velocity.h5
```

Run `summarize_mintpy_huber_full_scene.py` to regenerate the metrics and figures.
