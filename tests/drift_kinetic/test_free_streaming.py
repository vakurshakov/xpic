#!/usr/bin/env python3
"""Focused synthetic tests for free_streaming.py."""

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from free_streaming import (
    FreeStreamingError,
    complex_first_harmonic,
    free_streaming_prediction,
    load_config,
    main,
    run_analysis,
)


def _write_synthetic_run(root: Path, *, duration: int = 30,
                         conflicting_electric: bool = False,
                         extra_diagnostic: bool = False,
                         temperature_z_kev: float = 1.0) -> Path:
    """Write a complete, small config-plus-frame tree for integration tests."""
    run = root / "drift_kinetic_ringdown_ex0"
    directories = (run / "E", run / "negative" / "density",
                   run / "positive" / "density")
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    length_z = 2.0 * math.pi
    nz = 32
    diagnostics = [
        {"diagnostic": "FieldView", "field": "E", "component": "z",
         "out_dir": "E"},
        {"diagnostic": "DistributionMoment", "particles": "negative",
         "moment": "density", "out_dir": "negative/density"},
        {"diagnostic": "DistributionMoment", "particles": "positive",
         "moment": "density", "out_dir": "positive/density"},
    ]
    if extra_diagnostic:
        diagnostics.append({
            "diagnostic": "DistributionMoment", "particles": "negative",
            "moment": "current", "out_dir": "negative/current"})
    presets = [
        {"command": "SetMagneticField", "field": "B0", "field_axpy": "B",
         "setter": {"name": "SetUniformField", "value": [0.0, 0.0, 1.0]}},
        {"command": "SetElectricField", "field": "E",
         "setter": {"name": "SetUniformField", "value": [0.0, 0.0, 0.0]}},
        {"command": "SetParticles", "particles": "negative",
         "paired_with": "positive",
         "coordinate": {
             "name": "CoordinateInBoxQuietSineExactPaired",
             "min": [0.0, 0.0, 0.0], "max": [3.0, 3.0, length_z],
             "amplitude": [0.0, 0.0, 0.03],
             "wave_number": [0.0, 0.0, 1.0],
             "phase": [0.0, 0.0, 0.0]},
         "momentum": {"name": "MaxwellianVelocityQuiet"},
         "momentum_paired": {"name": "MaxwellianVelocityQuiet"}},
    ]
    if conflicting_electric:
        presets.append({
            "command": "SetElectricField", "field": "E",
            "setter": {"name": "SetUniformField", "value": [0.0, 0.0, 0.1]}})
    config = {
        "Simulation": "drift_kinetic",
        "OutputDirectory": str(run),
        "AnalyticTest": {
            "model": "neutral_free_streaming",
            "density_amplitude": 0.03,
            "mode_z": 1,
        },
        "Geometry": {
            "x": 3.0, "y": 3.0, "z": length_z, "t": float(duration),
            "dx": 1.0, "dy": 1.0, "dz": length_z / nz, "dt": 1.0,
            "diagnose_period": 1.0,
            "da_boundary_x": "DM_BOUNDARY_PERIODIC",
            "da_boundary_y": "DM_BOUNDARY_PERIODIC",
            "da_boundary_z": "DM_BOUNDARY_PERIODIC",
        },
        "Particles": [
            {"sort_name": "negative", "Np": 128, "n": 1.0, "q": -1.0,
             "m": 1.0, "Tx": 0.0, "Ty": 0.0, "Tz": temperature_z_kev,
             "coord_is_gc": True},
            {"sort_name": "positive", "Np": 128, "n": 1.0, "q": 1.0,
             "m": 1.0, "Tx": 0.0, "Ty": 0.0, "Tz": temperature_z_kev,
             "coord_is_gc": True},
        ],
        "Presets": presets,
        "Diagnostics": diagnostics,
    }
    (run / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")

    z = (np.arange(nz, dtype=float) + 0.5) * length_z / nz
    thermal_velocity = math.sqrt(temperature_z_kev / 511.0)
    cic_factor = float(np.sinc(1.0 / nz) ** 2)
    initial = -1j * 0.03 * cic_factor
    transverse = np.ones((1, 3, 3))
    for frame in range(duration + 1):
        harmonic = initial * math.exp(
            -0.5 * (thermal_velocity * frame) ** 2)
        profile = 1.0 + np.real(harmonic * np.exp(1j * z))
        density = (profile[:, None, None] * transverse).astype(np.float32)
        electric = np.zeros_like(density)
        name = f"{frame:04d}"
        density.tofile(run / "negative" / "density" / name)
        density.tofile(run / "positive" / "density" / name)
        electric.tofile(run / "E" / name)
    return run


class FreeStreamingTheoryTest(unittest.TestCase):
    def test_complex_harmonic_follows_maxwellian_ballistic_decay(self):
        nz = 128
        length = 2.0 * math.pi
        dz = length / nz
        k = 1.0
        thermal_velocity = math.sqrt(1.0 / 511.0)
        time = np.linspace(0.0, 45.0, 91)
        initial = 0.03 * np.exp(0.37j)
        expected = free_streaming_prediction(
            initial, time, k, thermal_velocity)

        z = (np.arange(nz, dtype=float) + 0.5) * dz
        recovered = []
        for harmonic in expected:
            profile = 1.0 + np.real(harmonic * np.exp(1j * k * z))
            recovered.append(complex_first_harmonic(profile, 1.0, k, dz))

        np.testing.assert_allclose(recovered, expected, rtol=2.0e-13,
                                   atol=2.0e-15)

    def test_prediction_preserves_phase_and_has_gaussian_envelope(self):
        time = np.array([0.0, 1.5, 4.0])
        initial = -0.012 + 0.027j
        k = 0.8
        thermal_velocity = 0.06
        result = free_streaming_prediction(initial, time, k, thermal_velocity)

        np.testing.assert_allclose(
            result / initial,
            np.exp(-0.5 * np.square(k * thermal_velocity * time)),
            rtol=1.0e-15, atol=1.0e-15)


class FreeStreamingEndToEndTest(unittest.TestCase):
    def test_valid_tree_passes_and_writes_all_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _write_synthetic_run(root)
            report_dir = root / "report"
            args = argparse.Namespace(
                run=str(run), out_dir=str(report_dir), check=True)

            with contextlib.redirect_stdout(io.StringIO()):
                return_code = run_analysis(args)

            self.assertEqual(return_code, 0)
            expected = {
                "free_streaming_results.json",
                "free_streaming_results.csv",
                "free_streaming.png",
            }
            self.assertTrue(expected.issubset(
                {path.name for path in report_dir.iterdir()}))
            report = json.loads(
                (report_dir / "free_streaming_results.json").read_text(
                    encoding="utf-8"))
            self.assertTrue(report["pass"])
            self.assertLessEqual(
                report["coverage"]["final_theory_fraction"], 0.5)

    def test_stale_extra_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _write_synthetic_run(Path(temporary))
            stale = np.zeros((32, 3, 3), dtype=np.float32)
            stale.tofile(run / "negative" / "density" / "9999")

            with self.assertRaisesRegex(
                    FreeStreamingError, "stale extra frame"):
                load_config(str(run))

    def test_short_run_is_rejected_by_decay_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _write_synthetic_run(root, duration=16)
            stderr = io.StringIO()
            argv = ["free_streaming.py", "--run", str(run),
                    "--out-dir", str(root / "report"), "--check"]

            with mock.patch.object(sys, "argv", argv), \
                    contextlib.redirect_stderr(stderr):
                return_code = main()

            self.assertEqual(return_code, 2)
            self.assertIn("coverage is too short", stderr.getvalue())

    def test_too_few_frames_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _write_synthetic_run(Path(temporary), duration=10)

            with self.assertRaisesRegex(
                    FreeStreamingError, "at least 16 diagnostic frames"):
                load_config(str(run))

    def test_duration_must_end_on_diagnostic_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _write_synthetic_run(Path(temporary), duration=29)
            config_path = run / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["Geometry"]["diagnose_period"] = 2.0
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with self.assertRaisesRegex(
                    FreeStreamingError, "must end on a diagnostic frame"):
                load_config(str(run))

    def test_conflicting_electric_writers_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _write_synthetic_run(
                Path(temporary), conflicting_electric=True)

            with self.assertRaisesRegex(
                    FreeStreamingError, "exactly one SetElectricField"):
                load_config(str(run))

    def test_extra_diagnostic_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _write_synthetic_run(
                Path(temporary), extra_diagnostic=True)

            with self.assertRaisesRegex(
                    FreeStreamingError, "exactly three diagnostics"):
                load_config(str(run))

    def test_non_negligible_subluminal_truncation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _write_synthetic_run(
                Path(temporary), temperature_z_kev=20.0)

            with self.assertRaisesRegex(
                    FreeStreamingError, r"6\*vT=.*>= 1"):
                load_config(str(run))


if __name__ == "__main__":
    unittest.main()
