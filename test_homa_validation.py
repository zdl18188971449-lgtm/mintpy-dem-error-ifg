import unittest

import numpy as np

import run_homa_statistical_validation as statistical
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


if __name__ == "__main__":
    unittest.main()
