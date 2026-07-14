#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("mintpy_dem_error_timeseries.py")
SPEC = importlib.util.spec_from_file_location("mintpy_dem_error_timeseries", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MintPyCompatibleEstimatorTest(unittest.TestCase):
    def synthetic_system(self):
        tbase = np.linspace(0.0, 2.0, 12, dtype=np.float64).reshape(-1, 1)
        geometry = np.asarray(
            [0.0, 0.0011, -0.0007, 0.0018, -0.0013, 0.0022,
             -0.0019, 0.0015, -0.0009, 0.0025, -0.0021, 0.0017],
            dtype=np.float64,
        ).reshape(-1, 1)
        deformation = np.column_stack((np.ones(12), tbase[:, 0], tbase[:, 0] ** 2))
        design = np.hstack((geometry, deformation))
        parameters = np.asarray(
            [[18.0, -7.0], [0.003, -0.004], [0.02, -0.01], [0.004, 0.002]],
            dtype=np.float64,
        )
        timeseries = design @ parameters
        return timeseries, design, tbase

    def test_matches_mintpy_on_clean_system(self):
        try:
            from mintpy.dem_error import estimate_dem_error
        except ImportError:
            self.skipTest("MintPy is not installed in this Python environment")

        timeseries, design, tbase = self.synthetic_system()
        expected = estimate_dem_error(timeseries, design, tbase)
        actual = MODULE.estimate_dem_error_huber(timeseries, design, tbase)
        for expected_array, actual_array in zip(expected, actual):
            np.testing.assert_allclose(actual_array, expected_array, rtol=1e-10, atol=1e-12)

    def test_huber_reduces_single_acquisition_outlier(self):
        from scipy import linalg

        clean, design, tbase = self.synthetic_system()
        contaminated = clean.copy()
        contaminated[7, :] += np.asarray([0.08, -0.06])
        ols_dem = linalg.lstsq(design, contaminated, cond=1e-8)[0][0]
        robust_dem = MODULE.estimate_dem_error_huber(
            contaminated,
            design,
            tbase,
            max_iterations=12,
        )[0]
        truth = np.asarray([18.0, -7.0])
        self.assertLess(np.mean(np.abs(robust_dem - truth)), np.mean(np.abs(ols_dem - truth)))

    def test_matches_mintpy_phase_velocity_on_clean_system(self):
        try:
            from mintpy.dem_error import estimate_dem_error
        except ImportError:
            self.skipTest("MintPy is not installed in this Python environment")

        timeseries, design, tbase = self.synthetic_system()
        expected = estimate_dem_error(
            timeseries,
            design,
            tbase,
            phase_velocity=True,
        )
        actual = MODULE.estimate_dem_error_huber(
            timeseries,
            design,
            tbase,
            phase_velocity=True,
        )
        for expected_array, actual_array in zip(expected, actual):
            np.testing.assert_allclose(actual_array, expected_array, rtol=1e-10, atol=1e-12)

    def test_excluded_date_is_not_used_for_estimation(self):
        clean, design, tbase = self.synthetic_system()
        contaminated = clean.copy()
        contaminated[5, :] += 2.0
        date_flag = np.ones(clean.shape[0], dtype=bool)
        date_flag[5] = False
        delta_z, corrected, residual = MODULE.estimate_dem_error_huber(
            contaminated,
            design,
            tbase,
            date_flag=date_flag,
        )
        np.testing.assert_allclose(delta_z, [18.0, -7.0], atol=1e-10)
        self.assertEqual(corrected.shape, clean.shape)
        self.assertEqual(residual.shape, clean.shape)
        self.assertGreater(np.min(np.abs(residual[5])), 1.9)

    def test_batched_pixelwise_solver_matches_scalar_huber(self):
        clean, design, tbase = self.synthetic_system()
        observations = np.repeat(clean[:, :1], 7, axis=1)
        geometry_scale = np.linspace(0.7, 1.3, observations.shape[1])
        geometry = design[:, :1] * geometry_scale[None, :]
        deformation = design[:, 1:]
        parameters = np.column_stack(
            (
                np.linspace(12.0, 20.0, observations.shape[1]),
                np.full(observations.shape[1], 0.003),
                np.linspace(0.01, 0.03, observations.shape[1]),
                np.linspace(-0.002, 0.004, observations.shape[1]),
            )
        )
        observations = np.empty((design.shape[0], observations.shape[1]))
        for pixel in range(observations.shape[1]):
            pixel_design = np.column_stack((geometry[:, pixel], deformation))
            observations[:, pixel] = pixel_design @ parameters[pixel]
        observations[7, 2:6] += np.linspace(0.03, 0.09, 4)

        expected = [[], [], []]
        for pixel in range(observations.shape[1]):
            pixel_design = np.column_stack((geometry[:, pixel], deformation))
            result = MODULE.estimate_dem_error_huber(
                observations[:, pixel],
                pixel_design,
                tbase,
                max_iterations=8,
            )
            expected[0].append(result[0][0])
            expected[1].append(result[1][:, 0])
            expected[2].append(result[2][:, 0])
        expected = (
            np.asarray(expected[0]),
            np.asarray(expected[1]).T,
            np.asarray(expected[2]).T,
        )
        actual = MODULE.estimate_dem_error_huber_batch(
            observations,
            geometry,
            deformation,
            tbase,
            max_iterations=8,
            pixel_batch_size=3,
        )
        for expected_array, actual_array in zip(expected, actual):
            np.testing.assert_allclose(actual_array, expected_array, rtol=2e-5, atol=2e-8)

    def test_batched_pixelwise_ols_matches_scalar_lstsq(self):
        rng = np.random.default_rng(19)
        _, design, tbase = self.synthetic_system()
        deformation = design[:, 1:]
        geometry = design[:, :1] * np.linspace(0.6, 1.4, 9)[None, :]
        observations = rng.normal(0.0, 0.02, (design.shape[0], geometry.shape[1]))
        expected_dem = []
        expected_corrected = []
        expected_residual = []
        for pixel in range(observations.shape[1]):
            pixel_design = np.column_stack((geometry[:, pixel], deformation))
            solution = np.linalg.lstsq(pixel_design, observations[:, pixel], rcond=1e-8)[0]
            expected_dem.append(solution[0])
            expected_corrected.append(observations[:, pixel] - pixel_design[:, 0] * solution[0])
            expected_residual.append(observations[:, pixel] - pixel_design @ solution)
        actual = MODULE.estimate_dem_error_huber_batch(
            observations,
            geometry,
            deformation,
            tbase,
            max_iterations=0,
            pixel_batch_size=4,
        )
        np.testing.assert_allclose(actual[0], expected_dem, rtol=2e-8, atol=2e-8)
        np.testing.assert_allclose(actual[1], np.asarray(expected_corrected).T, atol=2e-10)
        np.testing.assert_allclose(actual[2], np.asarray(expected_residual).T, atol=2e-10)


if __name__ == "__main__":
    unittest.main()
