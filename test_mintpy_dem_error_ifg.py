#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


SCRIPT = Path(__file__).with_name("mintpy_dem_error_ifg.py")
SPEC = importlib.util.spec_from_file_location("mintpy_dem_error_ifg", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DemErrorIfgramTest(unittest.TestCase):
    def setUp(self):
        self.date_pairs = [
            ("20200101", "20200201"),
            ("20200101", "20200301"),
            ("20200101", "20200401"),
            ("20200201", "20200301"),
            ("20200201", "20200401"),
            ("20200301", "20200401"),
        ]

    def synthetic_block(self):
        deformation, _ = MODULE.build_deformation_design(self.date_pairs, 1)
        bperp = np.asarray([120.0, -240.0, 310.0, -360.0, 190.0, 550.0])
        wavelength = 0.236
        inverse_geometry = 1.0 / (870000.0 * np.sin(np.deg2rad(39.0)))
        dem_coefficient = (-4.0 * np.pi / wavelength) * bperp[:, None] * inverse_geometry
        dem_coefficient = np.repeat(dem_coefficient, 12, axis=1)
        dem_error = np.linspace(-8.0, 9.0, 12)
        velocity = np.linspace(-0.02, 0.03, 12)
        phase = dem_coefficient * dem_error[None, :] + deformation @ velocity[None, :]
        return deformation, dem_coefficient, dem_error, phase

    def test_ols_recovers_exact_dem_error(self):
        deformation, coefficient, expected, phase = self.synthetic_block()
        solution, valid = MODULE.solve_model_block(
            phase,
            coefficient,
            deformation,
            np.ones(len(self.date_pairs), dtype=bool),
            np.ones_like(phase),
        )
        self.assertTrue(np.all(valid))
        np.testing.assert_allclose(solution[:, 0], expected, atol=1e-8)

    def test_huber_reduces_single_ifgram_outlier(self):
        deformation, coefficient, expected, phase = self.synthetic_block()
        phase = phase.copy()
        phase[2] += 0.8
        fit = np.ones(len(self.date_pairs), dtype=bool)
        weights = np.ones_like(phase)
        ols, _ = MODULE.solve_model_block(phase, coefficient, deformation, fit, weights)
        huber, _ = MODULE.solve_model_block(
            phase,
            coefficient,
            deformation,
            fit,
            weights,
            robust=True,
            huber_iterations=8,
        )
        ols_error = np.mean(np.abs(ols[:, 0] - expected))
        huber_error = np.mean(np.abs(huber[:, 0] - expected))
        self.assertLess(huber_error, ols_error)

    def test_adaptive_solver_selects_quadratic_deformation(self):
        quadratic_design, _ = MODULE.build_deformation_design(self.date_pairs, 2)
        bperp = np.asarray([120.0, -240.0, 310.0, -360.0, 190.0, 550.0])
        coefficient = (-4.0 * np.pi / 0.236) * bperp[:, None]
        coefficient /= 870000.0 * np.sin(np.deg2rad(39.0))
        coefficient = np.repeat(coefficient, 12, axis=1)
        expected = np.linspace(-8.0, 9.0, 12)
        velocity = np.linspace(-0.02, 0.03, 12)
        acceleration = np.linspace(-1.0, 1.0, 12)
        phase = coefficient * expected[None, :]
        phase += quadratic_design @ np.vstack((velocity, acceleration))

        solution, valid, information, uncertainty, order, labels = (
            MODULE.solve_adaptive_model_block(
                phase,
                coefficient,
                self.date_pairs,
                np.ones(len(self.date_pairs), dtype=bool),
                np.ones_like(phase),
                huber_iterations=8,
            )
        )
        self.assertTrue(np.all(valid))
        self.assertEqual(labels, ["velocity", "acceleration"])
        self.assertTrue(np.all(order == 2))
        self.assertTrue(np.all(information > 0))
        self.assertTrue(np.all(np.isfinite(uncertainty)))
        np.testing.assert_allclose(solution[:, 0], expected, atol=1e-8)

    def test_terrain_graph_reduces_noise_and_preserves_step(self):
        rng = np.random.default_rng(3)
        truth = np.zeros((20, 20), dtype=np.float64)
        truth[:, 10:] = 8.0
        terrain = np.zeros_like(truth)
        terrain[:, 10:] = 100.0
        initial = truth + rng.normal(0.0, 2.0, truth.shape)
        valid = np.ones_like(truth, dtype=bool)
        valid[8:12, 8:12] = False
        initial[~valid] = np.nan
        regularized, observability = MODULE.terrain_graph_regularize(
            initial,
            np.ones_like(truth),
            terrain,
            valid,
            regularization=8.0,
            iterations=40,
        )
        initial_rmse = np.sqrt(np.mean((initial[valid] - truth[valid]) ** 2))
        regularized_rmse = np.sqrt(
            np.mean((regularized[valid] - truth[valid]) ** 2)
        )
        self.assertLess(regularized_rmse, initial_rmse)
        self.assertTrue(np.all(np.isfinite(regularized[valid])))
        self.assertTrue(np.all(np.isnan(regularized[~valid])))
        self.assertLess(np.abs(np.nanmean(regularized[:, :8])), 0.5)
        self.assertGreater(np.nanmean(regularized[:, 12:]), 7.5)
        np.testing.assert_allclose(observability[valid], 1.0)

    def test_end_to_end_hdf_output(self):
        deformation, coefficient, expected, phase = self.synthetic_block()
        length, width = 3, 4
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stack_path = root / "ifgramStack.h5"
            geometry_path = root / "geometryRadar.h5"
            output_dir = root / "output"
            bperp = np.asarray([120.0, -240.0, 310.0, -360.0, 190.0, 550.0], np.float32)
            date_data = np.asarray(self.date_pairs, dtype="S8")

            with h5py.File(stack_path, "w") as file:
                file.attrs["WAVELENGTH"] = "0.236"
                file.attrs["LENGTH"] = str(length)
                file.attrs["WIDTH"] = str(width)
                file.attrs["FILE_TYPE"] = "ifgramStack"
                file.create_dataset("unwrapPhase", data=phase.reshape(6, length, width).astype(np.float32))
                file.create_dataset("coherence", data=np.full((6, length, width), 0.9, np.float32))
                file.create_dataset("date", data=date_data)
                file.create_dataset("bperp", data=bperp)
                file.create_dataset("dropIfgram", data=np.ones(6, dtype=bool))

            with h5py.File(geometry_path, "w") as file:
                file.create_dataset("incidenceAngle", data=np.full((length, width), 39.0, np.float32))
                file.create_dataset("slantRangeDistance", data=np.full((length, width), 870000.0, np.float32))

            rc = MODULE.main(
                [
                    str(stack_path),
                    "-g",
                    str(geometry_path),
                    "-o",
                    str(output_dir),
                    "--models",
                    "linear_ols",
                    "--block-rows",
                    "2",
                ]
            )
            self.assertEqual(rc, 0)
            corrected_path = output_dir / "ifgramStack_demErr_linear_ols.h5"
            component_path = output_dir / "demComponent_linear_ols.h5"
            self.assertTrue(corrected_path.exists())
            self.assertTrue(component_path.exists())
            with h5py.File(component_path, "r") as file:
                np.testing.assert_allclose(file["demError"][:].reshape(-1), expected, atol=2e-4)
                np.testing.assert_allclose(
                    file["demPhase"][:].reshape(6, -1),
                    coefficient * expected[None, :],
                    atol=2e-5,
                )
            with h5py.File(corrected_path, "r") as file:
                corrected = file["unwrapPhase"][:].reshape(6, -1)
                np.testing.assert_allclose(corrected, deformation @ np.linspace(-0.02, 0.03, 12)[None, :], atol=2e-5)
            try:
                from mintpy.objects import ifgramStack
            except ImportError:
                return
            stack = ifgramStack(str(corrected_path))
            stack.open(print_msg=False)
            self.assertEqual(stack.numIfgram, 6)


if __name__ == "__main__":
    unittest.main()
