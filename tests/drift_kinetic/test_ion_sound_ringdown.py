#!/usr/bin/env python3
"""Focused synthetic checks for ion_sound_ringdown.py."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ion_sound import MEC2_KEV, species_from_config
from ion_sound_ringdown import (
    RingdownError,
    SUITES,
    _comparison_config,
    _cross_run_result,
    _require_strict_push,
    fit_complex_ringdown,
    two_branch_model,
    validate_matching_configs,
)


class ModelVisualizationConfigTest(unittest.TestCase):
    def test_species_parser_prefers_parallel_temperature_with_legacy_fallback(self):
        modern = {
            "Particles": [
                {"sort_name": "electrons", "n": 1.0, "q": -1.0,
                 "m": 1.0, "Tx": 0.0, "Ty": 0.0, "Tz": 20.0},
                {"sort_name": "ions", "n": 1.0, "q": 1.0,
                 "m": 100.0, "Tx": 0.0, "Ty": 0.0, "Tz": 0.1},
            ],
        }
        electron, ion = species_from_config(modern)
        self.assertAlmostEqual(electron.T, 20.0 / MEC2_KEV)
        self.assertAlmostEqual(ion.T, 0.1 / MEC2_KEV)

        legacy = {
            "Particles": [
                {"sort_name": "electrons", "n": 1.0, "q": -1.0,
                 "m": 1.0, "T": 12.0},
                {"sort_name": "ions", "n": 1.0, "q": 1.0,
                 "m": 100.0, "T": 0.2},
            ],
        }
        electron, ion = species_from_config(legacy)
        self.assertAlmostEqual(electron.T, 12.0 / MEC2_KEV)
        self.assertAlmostEqual(ion.T, 0.2 / MEC2_KEV)


class TwoBranchFitTest(unittest.TestCase):
    def test_recovers_frequency_and_decrement_from_complex_signal(self):
        omega = 9.980432543835469e-3
        gamma = 6.105253504570409e-4
        period = 2.0 * math.pi / omega
        time = np.linspace(0.0, 5.0 * period, 501)
        a_plus = 0.014 * np.exp(0.31j)
        a_minus = 0.016 * np.exp(-0.27j)
        signal = two_branch_model(time, 0.0, omega, gamma, a_plus, a_minus)

        # Deterministic, zero-mean complex perturbation exercises a fit of raw
        # complex samples without making the test dependent on an RNG version.
        signal += 2.0e-7 * (
            np.sin(0.071 * np.arange(time.size))
            + 1j * np.cos(0.113 * np.arange(time.size)))
        result = fit_complex_ringdown(
            time, signal, omega, gamma, period, (0.5, 4.0),
            n_grid=31, n_passes=6)

        self.assertLess(abs(result["omega"] / omega - 1.0), 2.0e-3)
        self.assertLess(abs(result["gamma"] / gamma - 1.0), 2.0e-3)
        self.assertLess(result["relative_residual"], 2.0e-4)

    def test_rejects_zero_signal_before_fitting(self):
        omega = 1.0e-2
        period = 2.0 * math.pi / omega
        time = np.linspace(0.0, 5.0 * period, 101)
        with self.assertRaisesRegex(RingdownError, "zero signal"):
            fit_complex_ringdown(
                time, np.zeros(time.size, dtype=complex),
                omega, 6.0e-4, period, (0.5, 4.0))

    def test_light_window_recovers_frequency_and_decrement(self):
        omega = 9.980432543835469e-3
        gamma = 6.105253504570409e-4
        period = 2.0 * math.pi / omega
        time = np.linspace(0.0, 3.0 * period, 301)
        signal = two_branch_model(
            time, 0.0, omega, gamma,
            0.014 * np.exp(0.17j), 0.016 * np.exp(-0.21j))
        sample = np.arange(time.size)
        signal += 1.0e-7 * (
            np.sin(0.083 * sample) + 1j * np.cos(0.127 * sample))

        result = fit_complex_ringdown(
            time, signal, omega, gamma, period, (0.5, 2.5),
            n_grid=31, n_passes=6)

        self.assertLess(abs(result["omega"] / omega - 1.0), 2.0e-3)
        self.assertLess(abs(result["gamma"] / gamma - 1.0), 2.0e-3)
        self.assertLess(result["relative_residual"], 2.0e-4)


class SuitePolicyTest(unittest.TestCase):
    def test_comparison_config_removes_only_suite_parameter(self):
        config = {
            "OutputDirectory": "output/example",
            "DriftKineticPush": {
                "fail_on_terminal_nonconvergence": True,
            },
            "Geometry": {"z": 12.0, "dz": 0.375, "dt": 5.0},
            "Particles": [{"sort_name": "electrons", "Np": 512}],
        }

        publication = _comparison_config(config, SUITES["publication"])
        self.assertNotIn("OutputDirectory", publication)
        self.assertNotIn("Np", publication["Particles"][0])
        self.assertEqual(publication["Geometry"]["dz"], 0.375)
        self.assertTrue(publication["DriftKineticPush"]
                        ["fail_on_terminal_nonconvergence"])

        light = _comparison_config(config, SUITES["light"])
        self.assertNotIn("OutputDirectory", light)
        self.assertNotIn("dz", light["Geometry"])
        self.assertEqual(light["Particles"][0]["Np"], 512)

    def test_requires_strict_terminal_nonconvergence(self):
        path = Path("synthetic/config.json")
        _require_strict_push({
            "DriftKineticPush": {
                "fail_on_terminal_nonconvergence": True,
            },
        }, path)
        for config in ({},
                       {"DriftKineticPush": {}},
                       {"DriftKineticPush": {
                           "fail_on_terminal_nonconvergence": False,
                       }}):
            with self.subTest(config=config):
                with self.assertRaisesRegex(
                        RingdownError,
                        "fail_on_terminal_nonconvergence=true"):
                    _require_strict_push(config, path)

    def test_suite_grid_policy_rejects_old_publication_grid(self):
        strict = {
            "DriftKineticPush": {
                "fail_on_terminal_nonconvergence": True,
            },
        }
        old_run = SimpleNamespace(
            nz=128, directory=Path("old_ex1"),
            config_path=Path("old_ex1/config.json"), config=strict)
        with self.assertRaisesRegex(RingdownError,
                                    "publication ringdown requires Nz=32"):
            validate_matching_configs([old_run], SUITES["publication"])

        invalid_light = SimpleNamespace(
            nz=128, directory=Path("old_ex3"),
            config_path=Path("old_ex3/config.json"), config=strict)
        with self.assertRaisesRegex(RingdownError,
                                    "light ringdown allows only Nz=32/64"):
            validate_matching_configs([invalid_light], SUITES["light"])

    def test_light_cross_run_reports_grid_convergence(self):
        results = [
            {
                "particles_per_cell": 512,
                "nz": 64,
                "dz": 0.194,
                "fit": {"omega": 1.000e-2, "gamma": 6.00e-4},
            },
            {
                "particles_per_cell": 512,
                "nz": 32,
                "dz": 0.388,
                "fit": {"omega": 1.005e-2, "gamma": 6.30e-4},
            },
        ]

        result = _cross_run_result(results, SUITES["light"])

        self.assertTrue(result["pass"])
        self.assertEqual(result["comparison"], "grid")
        self.assertEqual(result["comparison_values"], [32, 64])
        self.assertTrue(result["checks"]["grid_points_z"])
        self.assertTrue(result["checks"]["common_particle_count"])


if __name__ == "__main__":
    unittest.main()
