# HOMA-DEM statistical and real-data validation

This validation separates three questions that must not be conflated:

1. **Synthetic truth accuracy:** repeated simulations quantify DEM-error RMSE against known truth.
2. **Real MintPy generalization:** held-out interferograms test prediction on observations excluded from estimation.
3. **External elevation agreement:** an independent Copernicus GLO-30 DSM tests corrected elevation after vertical-offset calibration on disjoint pixels.

## 1. Repeated simulations and error bars

Run 30 paired realizations. Every method receives exactly the same data for a given seed.

```bash
python run_homa_statistical_validation.py \
  -o homa_statistical_validation \
  --replicates 30 --seed-start 20260713 \
  --size 18 --workers 4 --dpi 600 --overwrite
```

The reported interval is a two-sided 95% t interval for the mean across seeds. The paired comparison table reports `other RMSE - HOMA RMSE`; positive values favor HOMA-DEM.

| Scenario | HOMA-DEM mean RMSE (95% CI) | Main published comparison | Interpretation |
|---|---:|---:|---|
| Static nonlinear | 0.0203 m (0.0181-0.0225) | Adaptive HT: 0.0375 m | HOMA is lower in all 30 paired realizations; paired mean reduction is 0.0172 m |
| Wrapped/cycle error | 0.0100 m (0.0090-0.0110) | Linear Huber: 0.00831 m | Linear Huber remains significantly better by 0.00169 m; HOMA is much more stable than the equivalent IGS implementation |
| Sparse DEM error | 0.1468 m (0.1280-0.1656) | Current graph: 1.0638 m; PGDC: 5.2609 m | HOMA lowers paired RMSE by 0.9170 m and 5.1141 m, respectively |
| Dynamic height | 0.00649 m (0.00426-0.00871) | Dynamic SBAS median: 0.0438 m | HOMA is accurate and has a much lighter failure tail; the Dynamic SBAS mean is inflated by rare extreme solutions |

### Ablation findings

- Removing IGS increases wrapped/cycle-error RMSE by about 8.9 times on a paired geometric-mean basis, confirming that wrapped-domain arbitration is essential in that scenario.
- Removing the dynamic branch increases dynamic-height RMSE by approximately three orders of magnitude.
- Removing ICA lowers sparse-scenario RMSE by about 23%, and removing PGDC lowers it by about 11%. These components are not universally beneficial and require stronger admission tests before a general superiority claim is defensible.
- Graph regularization has small effects in static/wrapped scenes and a modest positive effect in the sparse scenario.
- A validation-driven dynamic gate now rejects changes above 100 m by default, requires bounded before/after DEM estimates, at least nine connected pixels, and robust within-component amplitude consistency. This removed the static-scene tail failures without damaging the 16 m dynamic test.

Outputs:

- `repeated_metrics.csv`: every seed, scenario, method, accuracy, and runtime;
- `repeated_summary.csv`: mean, standard deviation, median, quartiles, and 95% CI;
- `paired_comparisons.csv`: paired differences and HOMA win rates;
- `homa_repeated_error_bars.*`: TGRS-style error-bar figure;
- `homa_ablation_heatmap.*`: module ablation figure.

Runtime columns from a multi-worker run are useful only as diagnostics because concurrent processes compete for CPU resources. Use `--workers 1` for defensible wall-clock comparisons.

## 2. Real MintPy network holdout

The included HFT473 `16 x 16` subset contains 26 interferograms. Five folds are formed deterministically; each training network remains connected. DEM error is estimated from the training interferograms only. A common cubic-plus-annual nuisance deformation model is then fitted on training residuals and evaluated on held-out interferograms.

```bash
python validate_homa_real_data.py \
  test_data/HFT473_16x16/ifgramStack.h5 \
  -g test_data/HFT473_16x16/geometryRadar.h5 \
  --mask test_data/HFT473_16x16/maskTempCoh.h5 \
  --mintpy-dem-reference /path/to/MintPy/demErr.h5 \
  -o homa_real_validation --folds 5 --dpi 600 --overwrite
```

| Method | Mean held-out phase RMSE (rad) | 95% CI |
|---|---:|---:|
| Linear Huber | 0.808 | 0.541-1.076 |
| Current graph | 0.811 | 0.543-1.079 |
| HOMA-DEM | 0.824 | 0.571-1.077 |
| Adaptive HT | 0.825 | 0.556-1.094 |
| No DEM correction | 0.831 | 0.542-1.120 |

HOMA improves slightly over no DEM correction but does not outperform Linear Huber on this small real subset. The overlapping confidence intervals and only five folds do not support a statistically significant ranking. Agreement with MintPy's full-stack `demErr.h5` is descriptive rather than independent validation because both estimates use the same interferogram network.

## 3. External Copernicus GLO-30 validation

The HFT473 radar subset is mapped to geographic coordinates with MintPy `geometryGeo.h5`. The external source used here is:

```text
Copernicus DSM GLO-30, tile N35E103
https://copernicus-dem-30m.s3.amazonaws.com/
Copernicus_DSM_COG_10_N35_00_E103_00_DEM/
Copernicus_DSM_COG_10_N35_00_E103_00_DEM.tif
```

The COG URL may be passed directly without downloading the complete tile:

```bash
python validate_homa_real_data.py \
  test_data/HFT473_16x16/ifgramStack.h5 \
  -g test_data/HFT473_16x16/geometryRadar.h5 \
  --mask test_data/HFT473_16x16/maskTempCoh.h5 \
  --mintpy-dem-reference /path/to/MintPy/demErr.h5 \
  --lookup /path/to/MintPy/inputs/geometryGeo.h5 \
  --external-dem "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N35_00_E103_00_DEM/Copernicus_DSM_COG_10_N35_00_E103_00_DEM.tif" \
  -o homa_real_validation --folds 5 --dpi 600 --overwrite
```

For each method, 128 checkerboard pixels estimate a robust relative vertical offset and a disjoint set of 128 pixels evaluates accuracy.

| Method | External RMSE (m) | External MAE (m) |
|---|---:|---:|
| No DEM correction | 11.55 | 9.16 |
| Current graph | 11.79 | 9.41 |
| HOMA-DEM | 12.04 | 9.40 |
| Linear Huber | 12.09 | 9.59 |
| Adaptive HT | 12.11 | 9.63 |
| MintPy demErr | 12.31 | 9.72 |

No correction is best in this external comparison. Therefore this subset provides no external evidence that DEM-error correction improves absolute elevation. Copernicus is a DSM rather than survey-grade bare-earth truth, and differences in acquisition epoch, vegetation/buildings, geolocation, spatial resolution, and vertical datum contribute to the 11-12 m residual. LiDAR or ICESat-2 photons with explicit quality filtering remain necessary for a strong publication claim.

## 4. Publication limits

- Thirty synthetic seeds support error bars but do not span all terrain, atmosphere, baseline-network, coherence, and unwrapping regimes.
- The real experiment uses one small ALOS subset. Add at least one different track/sensor and a larger spatial region.
- Five holdout folds are descriptive and correlated because they share acquisitions and pixels.
- The Copernicus comparison is independent but not survey-grade truth. Report it as external DSM agreement, not absolute DEM-error truth.
- Hyperparameters changed after observing validation failures must be tested on new, untouched scenes. The current HFT473 result must not be reused as evidence of out-of-sample tuning success.
