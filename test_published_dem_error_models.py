#!/usr/bin/env python3
from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import numpy as np

import published_dem_error_models as models


class PublishedDemErrorModelsTest(unittest.TestCase):
    def test_nonparametric_ica_recovers_baseline_component(self):
        rng = np.random.default_rng(10)
        num_acquisition = 9
        num_pixel = 500
        dates = [f"2020{index + 1:02d}01" for index in range(num_acquisition)]
        pairs = [
            (dates[left], dates[right])
            for left in range(num_acquisition)
            for right in range(left + 1, min(num_acquisition, left + 3))
        ]
        date_index = {date: index for index, date in enumerate(dates)}
        dem_error = rng.laplace(0.0, 7.0, num_pixel)
        dem_error -= np.mean(dem_error)
        deformation_source = rng.uniform(-2.0, 2.0, num_pixel)
        atmosphere_source = rng.standard_t(5, num_pixel)
        sequential_baseline = np.asarray([-0.35, 0.22, -0.48, 0.31, 0.55, -0.27, 0.42, -0.19])
        sequential_phase = sequential_baseline[:, None] * dem_error
        sequential_phase += np.linspace(-0.2, 0.3, 8)[:, None] * deformation_source
        sequential_phase += 0.15 * np.sin(np.arange(8) * 1.3)[:, None] * atmosphere_source
        acquisition_phase = np.vstack(
            (np.zeros(num_pixel), np.cumsum(sequential_phase, axis=0))
        )
        acquisition_coefficient = np.r_[0.0, np.cumsum(sequential_baseline)]
        phase = np.asarray(
            [
                acquisition_phase[date_index[slave]] - acquisition_phase[date_index[master]]
                for master, slave in pairs
            ]
        )
        coefficient = np.asarray(
            [
                acquisition_coefficient[date_index[slave]]
                - acquisition_coefficient[date_index[master]]
                for master, slave in pairs
            ]
        )[:, None]
        coefficient = np.repeat(coefficient, num_pixel, axis=1)

        result = models.nonparametric_ica_2019(phase, coefficient, pairs)

        rmse = np.sqrt(np.mean((result.dem_error - dem_error) ** 2))
        self.assertLess(rmse, 0.7)
        self.assertTrue(result.diagnostics["significant"])
        self.assertGreater(abs(result.diagnostics["baseline_correlation"]), 0.99)

    def test_adaptive_hypothesis_recovers_dem_with_nonlinear_deformation(self):
        rng = np.random.default_rng(4)
        dates = [f"2020{month:02d}01" for month in range(1, 10)]
        pairs = [
            (dates[left], dates[right])
            for left in range(len(dates))
            for right in range(left + 1, min(len(dates), left + 3))
        ]
        date_index = {date: index for index, date in enumerate(dates)}
        num_pixel = 80
        acquisition_baseline = np.linspace(-300.0, 350.0, len(dates))
        coefficient = np.asarray(
            [
                acquisition_baseline[date_index[slave]]
                - acquisition_baseline[date_index[master]]
                for master, slave in pairs
            ]
        )[:, None] * 0.002
        coefficient = np.repeat(coefficient, num_pixel, axis=1)
        dem_error = rng.normal(0.0, 8.0, num_pixel)
        years = np.arange(len(dates)) / 12.0
        velocity = rng.normal(0.0, 1.0, num_pixel)
        seasonal = rng.normal(0.0, 0.3, num_pixel)
        acquisition_phase = years[:, None] * velocity
        acquisition_phase += np.sin(2.0 * np.pi * years)[:, None] * seasonal
        phase = np.asarray(
            [
                acquisition_phase[date_index[slave]] - acquisition_phase[date_index[master]]
                for master, slave in pairs
            ]
        )
        phase += coefficient * dem_error

        result = models.adaptive_hypothesis_2021(
            phase, coefficient, pairs, group_span_years=0.6
        )

        rmse = np.sqrt(np.mean((result.dem_error - dem_error) ** 2))
        self.assertTrue(np.all(result.valid))
        self.assertLess(rmse, 0.8)

    def test_igs_equivalent_recovers_wrapped_phase_parameters(self):
        rng = np.random.default_rng(2)
        temporal = np.linspace(0.05, 1.2, 18)
        coefficient = rng.uniform(-0.12, 0.12, 18)[:, None]
        expected_dem = 37.5
        expected_velocity = 1.3
        wrapped = np.angle(
            np.exp(1j * (temporal[:, None] * expected_velocity + coefficient * expected_dem))
        )

        result = models.igs_cmaes_2021_equivalent(
            wrapped,
            coefficient,
            temporal,
            velocity_bounds=(-4.0, 4.0),
            dem_bounds=(-100.0, 100.0),
            maxiter=60,
        )

        self.assertAlmostEqual(result.dem_error[0], expected_dem, places=3)
        self.assertAlmostEqual(result.diagnostics["velocity"][0], expected_velocity, places=3)

    def test_pgdc_detects_and_estimates_sparse_dem_error(self):
        rng = np.random.default_rng(3)
        rows, columns, num_ifgram = 14, 14, 9
        truth = np.zeros((rows, columns))
        truth[5:9, 5:9] = 18.0
        baseline = np.asarray([-260, -180, -100, -40, 60, 130, 210, 290, 350.0])
        coefficient = np.broadcast_to(
            baseline[:, None, None] * 0.003, (num_ifgram, rows, columns)
        )
        phase = coefficient * truth + rng.normal(0.0, 0.015, coefficient.shape)
        wrapped = np.angle(np.exp(1j * phase))
        dates = [
            (datetime(2020, 1, 1) + timedelta(days=24 * index)).strftime("%Y%m%d")
            for index in range(num_ifgram + 1)
        ]
        pairs = list(zip(dates[:-1], dates[1:]))

        _, detected, _ = models.pgdc_detect_2025(
            wrapped, baseline, np.full(num_ifgram, 24.0), threshold=0.4
        )
        result = models.pgdc_estimate_2025(
            wrapped,
            coefficient,
            pairs,
            detected,
            dem_bounds=(-40.0, 40.0),
            dem_step=0.5,
            maximum_edge_length=6.0,
        )

        self.assertGreater(np.sum(detected & (truth != 0)), 0.7 * np.sum(truth != 0))
        self.assertGreater(np.mean(result.dem_error[truth != 0]), 7.0)
        self.assertLess(abs(np.mean(result.dem_error[(truth == 0) & result.valid])), 1.0)
        self.assertGreater(result.diagnostics["mean_tpc"], 0.95)

    def test_dynamic_height_search_recovers_change(self):
        dates = [f"2020{month:02d}01" for month in range(1, 11)]
        pairs = [
            (dates[left], dates[right])
            for left in range(len(dates))
            for right in range(left + 1, min(len(dates), left + 4))
        ]
        date_index = {date: index for index, date in enumerate(dates)}
        acquisition_baseline = np.asarray([-200, -120, 40, 180, -80, 260, 100, -250, 300, 20.0])
        coefficient = np.asarray(
            [
                acquisition_baseline[date_index[slave]]
                - acquisition_baseline[date_index[master]]
                for master, slave in pairs
            ]
        )[:, None] * 0.002
        temporal = np.asarray(
            [(date_index[slave] - date_index[master]) / 12.0 for master, slave in pairs]
        )
        before, after, change_index = 5.0, 17.0, 5
        phase = np.empty((len(pairs), 1))
        for index, (master, slave) in enumerate(pairs):
            master_index = date_index[master]
            slave_index = date_index[slave]
            if slave_index < change_index:
                dem_error = before
            elif master_index >= change_index:
                dem_error = after
            else:
                dem_error = 0.0
            phase[index, 0] = coefficient[index, 0] * dem_error + temporal[index] * 0.6

        result = models.dynamic_height_sbas_2025(phase, coefficient, pairs)

        self.assertEqual(result.diagnostics["change_index"][0], change_index)
        self.assertAlmostEqual(result.diagnostics["dem_error_before"][0], before, places=2)
        self.assertAlmostEqual(result.dem_error[0], after, places=2)

    def test_fractal_adaptation_reduces_spike_error(self):
        rng = np.random.default_rng(5)
        rows, columns = np.mgrid[:30, :30]
        truth = 5.0 * np.sin(columns / 5.0) + 2.0 * np.cos(rows / 6.0)
        initial = truth + rng.normal(0.0, 1.0, truth.shape)
        initial[10, 12] += 15.0
        magnitude_proxy = (0.2 + np.abs(np.gradient(truth, axis=1))) ** 4

        result = models.fractal_surface_regularize_2015_adapted(
            initial, magnitude_proxy
        )

        initial_rmse = np.sqrt(np.mean((initial - truth) ** 2))
        corrected_rmse = np.sqrt(np.mean((result.dem_error - truth) ** 2))
        self.assertLess(corrected_rmse, initial_rmse)
        self.assertEqual(result.diagnostics["status"], "adapted")

    def test_hybrid_combines_temporal_models_and_wrapped_fallback(self):
        rng = np.random.default_rng(14)
        size = 10
        dates = [
            (datetime(2020, 1, 1) + timedelta(days=36 * index)).strftime("%Y%m%d")
            for index in range(10)
        ]
        pairs = [
            (dates[left], dates[right])
            for left in range(len(dates))
            for right in range(left + 1, min(len(dates), left + 4))
        ]
        date_index = {date: index for index, date in enumerate(dates)}
        acquisition_baseline = rng.normal(0.0, 240.0, len(dates))
        coefficient_1d = np.asarray(
            [
                acquisition_baseline[date_index[slave]]
                - acquisition_baseline[date_index[master]]
                for master, slave in pairs
            ]
        ) * 0.0018
        coefficient = np.broadcast_to(
            coefficient_1d[:, None, None], (len(pairs), size, size)
        )
        rows, columns = np.mgrid[:size, :size]
        dem_error = 8.0 * np.sin(columns / 2.5) - 5.0 * np.cos(rows / 3.0)
        years = np.arange(len(dates)) * 36.0 / 365.25
        velocity = 0.6 * np.sin(rows / 3.0)
        acceleration = 0.3 * np.cos(columns / 3.0)
        acquisition_phase = years[:, None, None] * velocity
        acquisition_phase += 0.5 * years[:, None, None] ** 2 * acceleration
        phase = np.asarray(
            [
                acquisition_phase[date_index[slave]]
                - acquisition_phase[date_index[master]]
                for master, slave in pairs
            ]
        )
        phase += coefficient * dem_error
        phase += rng.normal(0.0, 0.01, phase.shape)
        unwrap_region = (rows - 5) ** 2 + (columns - 5) ** 2 <= 6
        for interferogram_index in (3, 11):
            phase[interferogram_index, unwrap_region] += 2.0 * np.pi
        wrapped = np.angle(np.exp(1j * phase))

        result = models.hybrid_optimal_2026(
            phase,
            coefficient,
            pairs,
            np.full_like(phase, 0.9),
            wrapped_phase=wrapped,
            terrain=np.zeros((size, size)),
            velocity_bounds=(-4.0, 4.0),
            dem_bounds=(-60.0, 60.0),
        )

        rmse = np.sqrt(np.nanmean((result.dem_error - dem_error) ** 2))
        self.assertLess(rmse, 0.25)
        self.assertGreater(result.diagnostics["igs"]["candidate_pixels"], 0)
        self.assertGreater(result.diagnostics["igs"]["arbitrated_pixels"], 0)
        self.assertFalse(np.any(result.diagnostics["dynamic_mask"]))

        ablated = models.hybrid_optimal_2026(
            phase,
            coefficient,
            pairs,
            np.full_like(phase, 0.9),
            wrapped_phase=wrapped,
            terrain=np.zeros((size, size)),
            velocity_bounds=(-4.0, 4.0),
            dem_bounds=(-60.0, 60.0),
            enable_ica=False,
            enable_dynamic=False,
            enable_pgdc=False,
            enable_igs=False,
            enable_graph=False,
        )
        self.assertEqual(
            ablated.diagnostics["ablation_flags"],
            {
                "ica": False,
                "dynamic": False,
                "pgdc": False,
                "igs": False,
                "graph": False,
            },
        )
        self.assertEqual(ablated.diagnostics["igs"]["candidate_pixels"], 0)
        self.assertFalse(np.any(ablated.diagnostics["dynamic_mask"]))
        self.assertFalse(np.any(ablated.diagnostics["pgdc_detected"]))
        self.assertFalse(np.any(ablated.diagnostics["graph_blend"]))

    def test_hybrid_selects_true_dynamic_height_region(self):
        rng = np.random.default_rng(18)
        size = 8
        dates = [f"2020{month:02d}01" for month in range(1, 11)]
        pairs = [
            (dates[left], dates[right])
            for left in range(len(dates))
            for right in range(left + 1, min(len(dates), left + 4))
        ]
        date_index = {date: index for index, date in enumerate(dates)}
        acquisition_baseline = rng.normal(0.0, 220.0, len(dates))
        acquisition_coefficient = 0.002 * acquisition_baseline
        coefficient_1d = np.asarray(
            [
                acquisition_coefficient[date_index[slave]]
                - acquisition_coefficient[date_index[master]]
                for master, slave in pairs
            ]
        )
        coefficient = np.broadcast_to(
            coefficient_1d[:, None, None], (len(pairs), size, size)
        )
        before = np.zeros((size, size))
        height_change = np.zeros_like(before)
        height_change[2:6, 3:6] = 14.0
        after = before + height_change
        change_index = 5
        phase = np.empty_like(coefficient)
        for pair_index, (master, slave) in enumerate(pairs):
            master_index = date_index[master]
            slave_index = date_index[slave]
            if slave_index < change_index:
                topographic = coefficient_1d[pair_index] * before
            elif master_index >= change_index:
                topographic = coefficient_1d[pair_index] * after
            else:
                topographic = (
                    acquisition_coefficient[slave_index] * after
                    - acquisition_coefficient[master_index] * before
                )
            phase[pair_index] = topographic
        phase += rng.normal(0.0, 0.005, phase.shape)

        result = models.hybrid_optimal_2026(
            phase,
            coefficient,
            pairs,
            np.full_like(phase, 0.92),
            terrain=np.zeros((size, size)),
            velocity_bounds=(-2.0, 2.0),
            dem_bounds=(-40.0, 40.0),
        )

        estimated_change = result.diagnostics["height_change"]
        rmse = np.sqrt(np.mean((estimated_change - height_change) ** 2))
        self.assertLess(rmse, 0.1)
        np.testing.assert_array_equal(
            result.diagnostics["dynamic_mask"], height_change != 0
        )
        self.assertTrue(
            np.all(
                result.diagnostics["change_index"][height_change != 0]
                == change_index
            )
        )

        bounded = models.hybrid_optimal_2026(
            phase,
            coefficient,
            pairs,
            np.full_like(phase, 0.92),
            terrain=np.zeros((size, size)),
            velocity_bounds=(-2.0, 2.0),
            dem_bounds=(-40.0, 40.0),
            maximum_height_change=10.0,
        )
        self.assertFalse(np.any(bounded.diagnostics["dynamic_mask"]))

    def test_dynamic_component_filter_rejects_small_and_inconsistent_regions(self):
        selected = np.zeros((5, 5), dtype=bool)
        selected[0, 0:2] = True
        selected[2, 1:5] = True
        change = np.zeros((5, 5), dtype=np.float64)
        change[0, 0:2] = 12.0
        change[2, 1:4] = 15.0
        change[2, 4] = 90.0

        filtered = models._filter_dynamic_components(
            selected.reshape(-1), change.reshape(-1), selected.shape, 3, 2.0
        ).reshape(selected.shape)

        self.assertFalse(np.any(filtered[0]))
        self.assertTrue(np.all(filtered[2, 1:4]))
        self.assertFalse(filtered[2, 4])


if __name__ == "__main__":
    unittest.main()
