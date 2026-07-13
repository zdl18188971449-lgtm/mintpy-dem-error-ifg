import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

import run_homa_statistical_validation as statistical
import run_homa_full_scene as full_scene
import validate_homa_real_data as real


class HomaValidationTest(unittest.TestCase):
    def test_mean_ci_and_summary_are_finite(self):
        rows = [
            {
                "scenario": "static_nonlinear",
                "method": "hybrid_optimal_2026",
                "label": "HOMA",
                "reproduction_status": "new_hybrid",
                "rmse_m": value,
                "runtime_s": 1.0,
            }
            for value in (0.1, 0.2, 0.3)
        ]
        summary = statistical._summarize(rows)[0]
        self.assertAlmostEqual(summary["mean_rmse_m"], 0.2)
        self.assertLess(summary["mean_ci95_low_m"], 0.2)
        self.assertGreater(summary["mean_ci95_high_m"], 0.2)

    def test_network_connectivity(self):
        self.assertTrue(real._network_connected([("a", "b"), ("b", "c")]))
        self.assertFalse(
            real._network_connected([("a", "b"), ("c", "d")])
        )

    def test_relative_map_metrics_remove_constant_offset(self):
        reference = np.asarray([[10.0, 12.0], [14.0, 16.0]])
        estimate = reference - 5.0
        metrics = real._relative_map_metrics(reference, estimate)
        self.assertAlmostEqual(metrics["offset_m"], 5.0)
        self.assertAlmostEqual(metrics["rmse_m"], 0.0)
        self.assertAlmostEqual(metrics["mae_m"], 0.0)
        self.assertAlmostEqual(metrics["correlation"], 1.0)

    def test_velocity_from_timeseries_recovers_linear_rate(self):
        dates = np.asarray([b"20200101", b"20210101", b"20220101"])
        values = np.asarray([0.0, 2.0, 4.0])[:, None, None]
        velocity = full_scene._velocity_from_timeseries(values, dates)
        self.assertAlmostEqual(float(velocity[0, 0]), 2.0, places=2)

    def test_full_scene_scale_recovers_known_fraction(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            stack_path = root / "stack.h5"
            geometry_path = root / "geometry.h5"
            mask_path = root / "mask.h5"
            component_path = root / "component.h5"
            incidence = np.full((2, 2), 35.0, dtype=np.float32)
            slant_range = np.full((2, 2), 800000.0, dtype=np.float32)
            bperp = np.asarray([100.0, 200.0], dtype=np.float32)
            wavelength = 0.24
            coefficient = (
                -4.0
                * np.pi
                / wavelength
                * bperp[:, None, None]
                / (slant_range * np.sin(np.deg2rad(incidence)))[None, :, :]
            )
            dem_error = np.asarray([[5.0, 8.0], [10.0, 12.0]], dtype=np.float32)
            with h5py.File(stack_path, "w") as stack:
                stack.attrs["WAVELENGTH"] = wavelength
                stack.create_dataset(
                    "date",
                    data=np.asarray(
                        [[b"20200101", b"20210101"], [b"20200101", b"20220101"]]
                    ),
                )
                stack.create_dataset("bperp", data=bperp)
                stack.create_dataset("coherence", data=np.ones((2, 2, 2)))
                stack.create_dataset("unwrapPhase", data=0.6 * coefficient * dem_error)
            with h5py.File(geometry_path, "w") as geometry:
                geometry.create_dataset("incidenceAngle", data=incidence)
                geometry.create_dataset("slantRangeDistance", data=slant_range)
            with h5py.File(mask_path, "w") as mask:
                mask.create_dataset("mask", data=np.ones((2, 2), dtype=bool))
            with h5py.File(component_path, "w") as component:
                component.create_dataset("demError", data=dem_error)

            metrics = full_scene._estimate_and_apply_dem_scale(
                stack_path,
                geometry_path,
                mask_path,
                None,
                component_path,
                0.5,
                2,
            )

            self.assertAlmostEqual(metrics["applied_global_scale"], 0.6, places=5)
            with h5py.File(component_path, "r") as component:
                np.testing.assert_allclose(component["demError"][:], 0.6 * dem_error)
                np.testing.assert_allclose(component["demErrorRaw"][:], dem_error)


if __name__ == "__main__":
    unittest.main()
