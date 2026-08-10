#!/usr/bin/env python3
"""Validate the analytic neutral free-streaming drift-kinetic test.

For a sinusoidal density perturbation with a Maxwellian parallel velocity
distribution and no force, the complex first harmonic obeys

    a1(t) = a1(0) exp[-0.5 (k vT t)^2],  vT = sqrt(Tz / (m mec^2)).

The two opposite-charge species in this test are loaded marker by marker with
identical coordinates and velocities.  Their density fields must therefore
remain equal, their deposited charge must cancel, and Ez must remain zero.

Every parameter and diagnostic path is read from the run's config.json.  The
reader rejects incomplete, non-contiguous, stale, or malformed output trees.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


MEC2_KEV = 511.0

# Acceptance criteria for the deliberately quiet, exactly paired ex0 test.
BALLISTIC_RELATIVE_L2_MAX = 0.03
BALLISTIC_MAX_ERROR_OVER_A0_MAX = 0.03
INITIAL_AMPLITUDE_RELATIVE_ERROR_MAX = 0.02
DENSITY_EQUALITY_RELATIVE_MAX = 1.0e-6
CHARGE_RELATIVE_MAX = 1.0e-6
EZ_ABSOLUTE_MAX = 1.0e-7

# A free-streaming comparison is meaningful only after a clearly measurable
# part of the initial mode has phase mixed away.  These are input-validity
# requirements rather than fit tolerances: a shorter run must be rejected
# instead of allowing a frozen density mode to pass the 3% error threshold.
MIN_DIAGNOSTIC_FRAMES = 16
FINAL_THEORY_FRACTION_MAX = 0.5

# MaxwellianVelocityQuiet rejects candidates with |v| >= 1.  This analysis uses
# the untruncated Gaussian characteristic function, so keep the one-dimensional
# six-sigma interval inside the subluminal domain.  Tx=Ty=0 is required below;
# hence only the parallel tail is relevant.  The omitted two-sided Gaussian
# probability is then below 2e-9.
THERMAL_SIGMA_MULTIPLIER = 6.0
THERMAL_SPEED_LIMIT = 1.0


class FreeStreamingError(RuntimeError):
    """The input cannot be analysed without guessing missing information."""


@dataclass(frozen=True)
class SpeciesConfig:
    role: str
    name: str
    n: float
    q: float
    m: float
    temperature_z_kev: float
    particles_per_cell: int
    density_dir: Path
    density_frames: dict[int, str]

    @property
    def thermal_velocity(self) -> float:
        return math.sqrt(self.temperature_z_kev / (self.m * MEC2_KEV))


@dataclass(frozen=True)
class RunConfig:
    argument: str
    directory: Path
    config_path: Path
    config: dict[str, Any]
    nx: int
    ny: int
    nz: int
    lx: float
    ly: float
    lz: float
    dx: float
    dy: float
    dz: float
    dt: float
    duration: float
    diagnose_period: float
    expected_last_frame: int
    mode: int
    requested_amplitude: float
    phase: float
    negative: SpeciesConfig
    positive: SpeciesConfig
    electric_dir: Path
    electric_frames: dict[int, str]

    @property
    def k(self) -> float:
        return 2.0 * math.pi * self.mode / self.lz


@dataclass
class SeriesData:
    time: np.ndarray
    negative_harmonic: np.ndarray
    positive_harmonic: np.ndarray
    negative_theory: np.ndarray
    positive_theory: np.ndarray
    density_difference_max: np.ndarray
    charge_rms_relative: np.ndarray
    charge_max_relative: np.ndarray
    electric_rms: np.ndarray
    electric_max: np.ndarray


def _mapping(parent: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise FreeStreamingError(f"{where}: required object '{key}' is missing")
    return value


def _number(parent: dict[str, Any], key: str, where: str,
            *, positive: bool = False, nonnegative: bool = False) -> float:
    if key not in parent:
        raise FreeStreamingError(f"{where}: required number '{key}' is missing")
    value = parent[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FreeStreamingError(f"{where}.{key}: expected a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise FreeStreamingError(f"{where}.{key}: value must be finite")
    if positive and result <= 0.0:
        raise FreeStreamingError(f"{where}.{key}: value must be positive")
    if nonnegative and result < 0.0:
        raise FreeStreamingError(f"{where}.{key}: value must be non-negative")
    return result


def _integer(value: float, where: str, *, positive: bool = False) -> int:
    result = int(round(value))
    if not math.isclose(value, result, rel_tol=0.0, abs_tol=1.0e-9):
        raise FreeStreamingError(f"{where}: expected an integer, got {value!r}")
    if positive and result <= 0:
        raise FreeStreamingError(f"{where}: value must be positive")
    return result


def _vector3(parent: dict[str, Any], key: str, where: str) \
        -> tuple[float, float, float]:
    value = parent.get(key)
    if not isinstance(value, list) or len(value) != 3:
        raise FreeStreamingError(f"{where}.{key}: expected a three-element array")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise FreeStreamingError(
                f"{where}.{key}[{index}]: expected a JSON number")
        number = float(item)
        if not math.isfinite(number):
            raise FreeStreamingError(
                f"{where}.{key}[{index}]: value must be finite")
        result.append(number)
    return tuple(result)  # type: ignore[return-value]


def _close(left: float, right: float, *, scale: float = 1.0) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-11,
                        abs_tol=1.0e-12 * max(1.0, scale))


def resolve_run(argument: str) -> tuple[Path, Path]:
    supplied = Path(argument).expanduser()
    if supplied.is_file() and supplied.name == "config.json":
        return supplied.parent.resolve(), supplied.resolve()
    if supplied.is_dir() and (supplied / "config.json").is_file():
        directory = supplied.resolve()
        return directory, directory / "config.json"
    raise FreeStreamingError(
        f"--run must be a directory containing config.json or the config.json "
        f"itself: {argument}")


def _numeric_frames(directory: Path, expected_bytes: int,
                    expected_last: int, description: str) -> dict[int, str]:
    if not directory.is_dir():
        raise FreeStreamingError(f"{description} directory not found: {directory}")
    result: dict[int, str] = {}
    for entry in directory.iterdir():
        if not entry.is_file() or not entry.name.isdigit():
            continue
        index = int(entry.name)
        if index in result:
            raise FreeStreamingError(
                f"{directory}: duplicate numeric frame index {index}")
        size = entry.stat().st_size
        if size != expected_bytes:
            raise FreeStreamingError(
                f"{entry}: expected {expected_bytes} bytes, found {size}")
        result[index] = entry.name

    expected = set(range(expected_last + 1))
    actual = set(result)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            preview = ", ".join(str(value) for value in missing[:8])
            details.append("missing frame(s) " + preview
                           + (" ..." if len(missing) > 8 else ""))
        if extra:
            preview = ", ".join(str(value) for value in extra[:8])
            details.append("stale extra frame(s) " + preview
                           + (" ..." if len(extra) > 8 else ""))
        raise FreeStreamingError(f"{directory}: " + "; ".join(details))
    return result


def _diagnostic_directory(config: dict[str, Any], run_dir: Path,
                          *, diagnostic_name: str, description: str,
                          particles: str | None = None,
                          moment: str | None = None,
                          field: str | None = None,
                          component: str | None = None) -> Path:
    diagnostics = config.get("Diagnostics")
    if not isinstance(diagnostics, list):
        raise FreeStreamingError("config: required Diagnostics array is missing")
    matches = []
    for item in diagnostics:
        if not isinstance(item, dict) or item.get("diagnostic") != diagnostic_name:
            continue
        if particles is not None and item.get("particles") != particles:
            continue
        if moment is not None and item.get("moment") != moment:
            continue
        if field is not None and item.get("field") != field:
            continue
        if component is not None and item.get("component") != component:
            continue
        matches.append(item)
    if len(matches) != 1:
        raise FreeStreamingError(
            f"config: require exactly one {description} diagnostic")
    diagnostic = matches[0]
    out_dir = diagnostic.get("out_dir")
    if not isinstance(out_dir, str) or not out_dir:
        raise FreeStreamingError(
            f"config: {description} diagnostic needs explicit out_dir")
    relative = Path(out_dir)
    if relative.is_absolute() or ".." in relative.parts:
        raise FreeStreamingError(
            f"config: {description} out_dir must stay inside the run directory")
    if "diagnose_period" in diagnostic:
        raise FreeStreamingError(
            f"config: per-diagnostic period is unsupported for {description}")
    return run_dir / relative


def _parse_species(config: dict[str, Any], run_dir: Path, expected_last: int,
                   density_bytes: int) -> tuple[SpeciesConfig, SpeciesConfig]:
    particles = config.get("Particles")
    if not isinstance(particles, list) or len(particles) != 2:
        raise FreeStreamingError("config: exactly two particle species are required")

    parsed: list[dict[str, Any]] = []
    for index, item in enumerate(particles):
        where = f"config.Particles[{index}]"
        if not isinstance(item, dict):
            raise FreeStreamingError(f"{where}: expected an object")
        name = item.get("sort_name")
        if not isinstance(name, str) or not name:
            raise FreeStreamingError(f"{where}.sort_name: required non-empty string")
        tx = _number(item, "Tx", where, nonnegative=True)
        ty = _number(item, "Ty", where, nonnegative=True)
        if not _close(tx, 0.0) or not _close(ty, 0.0):
            raise FreeStreamingError(f"{where}: free streaming requires Tx=Ty=0")
        if item.get("coord_is_gc") is not True:
            raise FreeStreamingError(
                f"{where}: free streaming requires coord_is_gc=true")
        parsed.append({
            "name": name,
            "n": _number(item, "n", where, positive=True),
            "q": _number(item, "q", where),
            "m": _number(item, "m", where, positive=True),
            "temperature": _number(item, "Tz", where, positive=True),
            "np": _integer(_number(item, "Np", where, positive=True),
                           f"{where}.Np", positive=True),
        })

    negatives = [item for item in parsed if item["q"] < 0.0]
    positives = [item for item in parsed if item["q"] > 0.0]
    if len(negatives) != 1 or len(positives) != 1:
        raise FreeStreamingError(
            "config: require exactly one q<0 and one q>0 species")
    negative_raw, positive_raw = negatives[0], positives[0]

    for key, label in (("n", "density"), ("m", "mass"),
                       ("temperature", "Tz"), ("np", "Np")):
        if not _close(float(negative_raw[key]), float(positive_raw[key]),
                      scale=float(negative_raw[key])):
            raise FreeStreamingError(
                f"config: paired species must have equal {label}")
    if not _close(float(negative_raw["q"]), -float(positive_raw["q"]),
                  scale=abs(float(negative_raw["q"]))):
        raise FreeStreamingError(
            "config: paired species charges must be equal and opposite")

    result: list[SpeciesConfig] = []
    for role, raw in (("negative", negative_raw), ("positive", positive_raw)):
        density_dir = _diagnostic_directory(
            config, run_dir, diagnostic_name="DistributionMoment",
            description=f"{raw['name']} density", particles=str(raw["name"]),
            moment="density")
        frames = _numeric_frames(
            density_dir, density_bytes, expected_last,
            f"{raw['name']} density")
        result.append(SpeciesConfig(
            role=role, name=str(raw["name"]), n=float(raw["n"]),
            q=float(raw["q"]), m=float(raw["m"]),
            temperature_z_kev=float(raw["temperature"]),
            particles_per_cell=int(raw["np"]), density_dir=density_dir,
            density_frames=frames))
    return result[0], result[1]


def _validate_initial_conditions(config: dict[str, Any], negative: str,
                                 positive: str, lengths: tuple[float, float, float],
                                 analytic_amplitude: float,
                                 analytic_mode: int) -> float:
    presets = config.get("Presets")
    if not isinstance(presets, list):
        raise FreeStreamingError("config: required Presets array is missing")
    species = {negative, positive}
    pairs: list[tuple[int, dict[str, Any]]] = []
    for index, preset in enumerate(presets):
        if not isinstance(preset, dict) or preset.get("command") != "SetParticles":
            continue
        if {preset.get("particles"), preset.get("paired_with")} == species:
            pairs.append((index, preset))
        elif preset.get("particles") in species or preset.get("paired_with") in species:
            raise FreeStreamingError(
                "config: species must be loaded by one shared paired preset")
    if len(pairs) != 1:
        raise FreeStreamingError(
            "config: require exactly one paired SetParticles preset")

    index, pair = pairs[0]
    where = f"config.Presets[{index}]"
    coordinate = _mapping(pair, "coordinate", where)
    momentum = _mapping(pair, "momentum", where)
    momentum_paired = _mapping(pair, "momentum_paired", where)
    if coordinate.get("name") != "CoordinateInBoxQuietSineExactPaired":
        raise FreeStreamingError(
            f"{where}: require CoordinateInBoxQuietSineExactPaired")
    for label, block in (("momentum", momentum),
                         ("momentum_paired", momentum_paired)):
        if block.get("name") != "MaxwellianVelocityQuiet":
            raise FreeStreamingError(
                f"{where}.{label}: require MaxwellianVelocityQuiet")

    minimum = _vector3(coordinate, "min", f"{where}.coordinate")
    maximum = _vector3(coordinate, "max", f"{where}.coordinate")
    amplitude = _vector3(coordinate, "amplitude", f"{where}.coordinate")
    wave_number = _vector3(coordinate, "wave_number", f"{where}.coordinate")
    phase = _vector3(coordinate, "phase", f"{where}.coordinate")
    if any(not _close(value, 0.0) for value in minimum):
        raise FreeStreamingError(f"{where}: coordinate min must be the box origin")
    if any(not _close(maximum[i], lengths[i], scale=lengths[i]) for i in range(3)):
        raise FreeStreamingError(f"{where}: coordinate max must match Geometry")
    if any(not _close(value, 0.0)
           for value in (*amplitude[:2], *wave_number[:2], *phase[:2])):
        raise FreeStreamingError(f"{where}: perturbation must be z-only")
    mode = _integer(wave_number[2], f"{where}.coordinate.wave_number[2]",
                    positive=True)
    if mode != analytic_mode:
        raise FreeStreamingError(
            f"{where}: coordinate mode does not match AnalyticTest.mode_z")
    if not _close(amplitude[2], analytic_amplitude,
                  scale=analytic_amplitude):
        raise FreeStreamingError(
            f"{where}: coordinate amplitude does not match AnalyticTest")

    electric_writers = [
        (index, preset) for index, preset in enumerate(presets)
        if isinstance(preset, dict)
        and preset.get("command") == "SetElectricField"
        and preset.get("field") == "E"
    ]
    if len(electric_writers) != 1:
        raise FreeStreamingError(
            "config: require exactly one SetElectricField writer for E")
    electric_index, electric = electric_writers[0]
    electric_where = f"config.Presets[{electric_index}]"
    electric_setter = _mapping(electric, "setter", electric_where)
    if electric_setter.get("name") != "SetUniformField":
        raise FreeStreamingError(
            f"{electric_where}: E must use SetUniformField")
    electric_value = _vector3(
        electric_setter, "value", f"{electric_where}.setter")
    if not all(_close(item, 0.0) for item in electric_value):
        raise FreeStreamingError(
            f"{electric_where}: require exact uniform E(t=0)=(0,0,0)")

    magnetic_writers = [
        (index, preset) for index, preset in enumerate(presets)
        if isinstance(preset, dict)
        and preset.get("command") == "SetMagneticField"
        and preset.get("field") == "B0"
    ]
    if len(magnetic_writers) != 1:
        raise FreeStreamingError(
            "config: require exactly one SetMagneticField writer for B0")
    magnetic_index, magnetic = magnetic_writers[0]
    magnetic_where = f"config.Presets[{magnetic_index}]"
    if magnetic.get("field_axpy") != "B":
        raise FreeStreamingError(
            f"{magnetic_where}: B0 must set field_axpy='B'")
    magnetic_setter = _mapping(magnetic, "setter", magnetic_where)
    if magnetic_setter.get("name") != "SetUniformField":
        raise FreeStreamingError(
            f"{magnetic_where}: B0 must use SetUniformField")
    magnetic_value = _vector3(
        magnetic_setter, "value", f"{magnetic_where}.setter")
    if not all(_close(value, expected) for value, expected in
               zip(magnetic_value, (0.0, 0.0, 1.0))):
        raise FreeStreamingError(
            f"{magnetic_where}: require exact uniform B0=(0,0,1)")
    return phase[2]


def load_config(argument: str) -> RunConfig:
    run_dir, config_path = resolve_run(argument)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise FreeStreamingError(f"cannot read {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise FreeStreamingError(f"{config_path}: root JSON value must be an object")
    if config.get("Simulation") != "drift_kinetic":
        raise FreeStreamingError("config.Simulation must be 'drift_kinetic'")
    diagnostics = config.get("Diagnostics")
    if not isinstance(diagnostics, list) or len(diagnostics) != 3:
        raise FreeStreamingError(
            "config: require exactly three diagnostics: Ez and two densities")

    analytic = _mapping(config, "AnalyticTest", "config")
    if analytic.get("model") != "neutral_free_streaming":
        raise FreeStreamingError(
            "config.AnalyticTest.model must be 'neutral_free_streaming'")
    requested_amplitude = _number(
        analytic, "density_amplitude", "config.AnalyticTest", positive=True)
    mode = _integer(_number(analytic, "mode_z", "config.AnalyticTest",
                            positive=True),
                    "config.AnalyticTest.mode_z", positive=True)

    geometry = _mapping(config, "Geometry", "config")
    lx = _number(geometry, "x", "config.Geometry", positive=True)
    ly = _number(geometry, "y", "config.Geometry", positive=True)
    lz = _number(geometry, "z", "config.Geometry", positive=True)
    duration = _number(geometry, "t", "config.Geometry", positive=True)
    dx = _number(geometry, "dx", "config.Geometry", positive=True)
    dy = _number(geometry, "dy", "config.Geometry", positive=True)
    dz = _number(geometry, "dz", "config.Geometry", positive=True)
    dt = _number(geometry, "dt", "config.Geometry", positive=True)
    diagnose_period = _number(
        geometry, "diagnose_period", "config.Geometry", positive=True)
    nx = _integer(lx / dx, "config.Geometry.x/dx", positive=True)
    ny = _integer(ly / dy, "config.Geometry.y/dy", positive=True)
    nz = _integer(lz / dz, "config.Geometry.z/dz", positive=True)
    nt = _integer(duration / dt, "config.Geometry.t/dt", positive=True)
    diagnostic_stride = _integer(
        diagnose_period / dt, "config.Geometry.diagnose_period/dt",
        positive=True)
    if nt % diagnostic_stride != 0:
        raise FreeStreamingError(
            "config: Geometry.t must end on a diagnostic frame")
    expected_last = nt // diagnostic_stride
    frame_count = expected_last + 1
    if frame_count < MIN_DIAGNOSTIC_FRAMES:
        raise FreeStreamingError(
            f"config: free-streaming coverage requires at least "
            f"{MIN_DIAGNOSTIC_FRAMES} diagnostic frames; found {frame_count}")
    for axis in "xyz":
        if geometry.get(f"da_boundary_{axis}") != "DM_BOUNDARY_PERIODIC":
            raise FreeStreamingError(
                f"config.Geometry.da_boundary_{axis} must be periodic")
    if mode >= nz // 2:
        raise FreeStreamingError("config: mode_z must lie below the Nyquist mode")

    cells = nx * ny * nz
    negative, positive = _parse_species(
        config, run_dir, expected_last, cells * 4)
    for species in (negative, positive):
        six_sigma_speed = THERMAL_SIGMA_MULTIPLIER * species.thermal_velocity
        if not six_sigma_speed < THERMAL_SPEED_LIMIT:
            raise FreeStreamingError(
                f"config: {species.name} has "
                f"{THERMAL_SIGMA_MULTIPLIER:g}*vT={six_sigma_speed:.6g} >= 1; "
                "the untruncated Gaussian free-streaming formula is invalid")

    last_diagnostic_time = expected_last * diagnose_period
    final_theory_fraction = math.exp(
        -0.5 * (2.0 * math.pi * mode / lz
                * negative.thermal_velocity * last_diagnostic_time) ** 2)
    if final_theory_fraction > FINAL_THEORY_FRACTION_MAX:
        raise FreeStreamingError(
            f"config: free-streaming coverage is too short; expected final "
            f"mode fraction {final_theory_fraction:.6g} exceeds "
            f"{FINAL_THEORY_FRACTION_MAX:g}")

    phase = _validate_initial_conditions(
        config, negative.name, positive.name, (lx, ly, lz),
        requested_amplitude, mode)

    electric_dir = _diagnostic_directory(
        config, run_dir, diagnostic_name="FieldView", description="Ez",
        field="E", component="z")
    electric_frames = _numeric_frames(
        electric_dir, cells * 4, expected_last, "Ez")

    return RunConfig(
        argument=argument, directory=run_dir, config_path=config_path,
        config=config, nx=nx, ny=ny, nz=nz, lx=lx, ly=ly, lz=lz,
        dx=dx, dy=dy, dz=dz, dt=dt, duration=duration,
        diagnose_period=diagnose_period, expected_last_frame=expected_last,
        mode=mode, requested_amplitude=requested_amplitude, phase=phase,
        negative=negative, positive=positive, electric_dir=electric_dir,
        electric_frames=electric_frames)


def complex_first_harmonic(profile: np.ndarray, background_density: float,
                           k: float, dz: float) -> complex:
    """Return 2 <(n/n0-1) exp(-ikz)> on cell-centred z samples."""
    values = np.asarray(profile, dtype=float)
    if values.ndim != 1 or values.size < 3:
        raise FreeStreamingError("density profile must be a one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise FreeStreamingError("density profile contains non-finite values")
    if not (background_density > 0.0 and math.isfinite(background_density)):
        raise FreeStreamingError("background density must be finite and positive")
    z = (np.arange(values.size, dtype=float) + 0.5) * dz
    return complex(2.0 * np.mean(
        (values / background_density - 1.0) * np.exp(-1j * k * z)))


def free_streaming_prediction(a0: complex, time: np.ndarray,
                              k: float, thermal_velocity: float) -> np.ndarray:
    """Analytic Maxwellian ballistic decay of one complex density harmonic."""
    values = np.asarray(time, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise FreeStreamingError("time must be a finite one-dimensional array")
    if not (math.isfinite(k) and k > 0.0):
        raise FreeStreamingError("k must be finite and positive")
    if not (math.isfinite(thermal_velocity) and thermal_velocity > 0.0):
        raise FreeStreamingError("thermal velocity must be finite and positive")
    return complex(a0) * np.exp(-0.5 * np.square(k * thermal_velocity * values))


def _read_scalar_frame(path: Path, cells: int, shape: tuple[int, int, int]) \
        -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32, count=cells)
    if data.size != cells or not np.all(np.isfinite(data)):
        raise FreeStreamingError(
            f"{path}: frame changed while reading or contains non-finite values")
    return data.reshape(shape)


def load_series(run: RunConfig) -> SeriesData:
    cells = run.nx * run.ny * run.nz
    shape = (run.nz, run.ny, run.nx)
    times: list[float] = []
    negative_harmonic: list[complex] = []
    positive_harmonic: list[complex] = []
    density_difference_max: list[float] = []
    charge_rms: list[float] = []
    charge_max: list[float] = []
    electric_rms: list[float] = []
    electric_max: list[float] = []
    charge_scale = abs(run.negative.q) * run.negative.n \
        + abs(run.positive.q) * run.positive.n

    for index in range(run.expected_last_frame + 1):
        negative = _read_scalar_frame(
            run.negative.density_dir / run.negative.density_frames[index],
            cells, shape)
        positive = _read_scalar_frame(
            run.positive.density_dir / run.positive.density_frames[index],
            cells, shape)
        electric = _read_scalar_frame(
            run.electric_dir / run.electric_frames[index], cells, shape)

        negative_relative = negative / run.negative.n
        positive_relative = positive / run.positive.n
        density_difference_max.append(float(np.max(
            np.abs(negative_relative - positive_relative))))
        charge = run.negative.q * negative + run.positive.q * positive
        charge_rms.append(float(np.sqrt(np.mean(np.square(charge))) / charge_scale))
        charge_max.append(float(np.max(np.abs(charge)) / charge_scale))
        electric_rms.append(float(np.sqrt(np.mean(np.square(electric)))))
        electric_max.append(float(np.max(np.abs(electric))))

        negative_profile = negative.mean(axis=(1, 2))
        positive_profile = positive.mean(axis=(1, 2))
        negative_harmonic.append(complex_first_harmonic(
            negative_profile, run.negative.n, run.k, run.dz))
        positive_harmonic.append(complex_first_harmonic(
            positive_profile, run.positive.n, run.k, run.dz))
        times.append(index * run.diagnose_period)

    time = np.asarray(times, dtype=float)
    negative_array = np.asarray(negative_harmonic, dtype=complex)
    positive_array = np.asarray(positive_harmonic, dtype=complex)
    if abs(negative_array[0]) <= 1.0e-12 or abs(positive_array[0]) <= 1.0e-12:
        raise FreeStreamingError("initial first density harmonic is zero")
    return SeriesData(
        time=time,
        negative_harmonic=negative_array,
        positive_harmonic=positive_array,
        negative_theory=free_streaming_prediction(
            negative_array[0], time, run.k, run.negative.thermal_velocity),
        positive_theory=free_streaming_prediction(
            positive_array[0], time, run.k, run.positive.thermal_velocity),
        density_difference_max=np.asarray(density_difference_max),
        charge_rms_relative=np.asarray(charge_rms),
        charge_max_relative=np.asarray(charge_max),
        electric_rms=np.asarray(electric_rms),
        electric_max=np.asarray(electric_max))


def _complex_json(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag),
            "abs": float(abs(value))}


def _species_result(species: SpeciesConfig, harmonic: np.ndarray,
                    theory: np.ndarray) -> dict[str, Any]:
    error = harmonic - theory
    relative_l2 = float(np.linalg.norm(error) / np.linalg.norm(theory))
    max_error_over_a0 = float(np.max(np.abs(error)) / abs(harmonic[0]))
    checks = {
        "relative_l2": relative_l2 <= BALLISTIC_RELATIVE_L2_MAX,
        "max_error_over_a0":
            max_error_over_a0 <= BALLISTIC_MAX_ERROR_OVER_A0_MAX,
    }
    return {
        "role": species.role,
        "name": species.name,
        "particles_per_cell": species.particles_per_cell,
        "mass": species.m,
        "temperature_z_kev": species.temperature_z_kev,
        "thermal_velocity": species.thermal_velocity,
        "initial_harmonic": _complex_json(harmonic[0]),
        "final_harmonic": _complex_json(harmonic[-1]),
        "relative_l2_error": relative_l2,
        "max_error_over_a0": max_error_over_a0,
        "checks": checks,
        "pass": all(checks.values()),
    }


def analyse(run: RunConfig, series: SeriesData) -> dict[str, Any]:
    negative = _species_result(
        run.negative, series.negative_harmonic, series.negative_theory)
    positive = _species_result(
        run.positive, series.positive_harmonic, series.positive_theory)

    cic_factor = float(np.sinc(run.mode / run.nz) ** 2)
    expected_initial = run.requested_amplitude * cic_factor
    measured_initial = float(0.5 * (abs(series.negative_harmonic[0])
                                    + abs(series.positive_harmonic[0])))
    initial_amplitude_error = float(
        abs(measured_initial / expected_initial - 1.0))
    initial_check = bool(
        initial_amplitude_error <= INITIAL_AMPLITUDE_RELATIVE_ERROR_MAX)

    mode_difference = np.abs(
        series.negative_harmonic - series.positive_harmonic)
    mode_scale = max(abs(series.negative_harmonic[0]),
                     abs(series.positive_harmonic[0]))
    mode_difference_relative = float(np.max(mode_difference) / mode_scale)
    density_difference = float(np.max(series.density_difference_max))
    charge_rms = float(np.max(series.charge_rms_relative))
    charge_max = float(np.max(series.charge_max_relative))
    neutral_checks = {
        "density_fields_equal": density_difference <= DENSITY_EQUALITY_RELATIVE_MAX,
        "first_harmonics_equal":
            mode_difference_relative <= DENSITY_EQUALITY_RELATIVE_MAX,
        "charge_rms": charge_rms <= CHARGE_RELATIVE_MAX,
        "charge_max": charge_max <= CHARGE_RELATIVE_MAX,
    }
    electric_rms = float(np.max(series.electric_rms))
    electric_max = float(np.max(series.electric_max))
    electric_checks = {
        "ez_rms": electric_rms <= EZ_ABSOLUTE_MAX,
        "ez_max": electric_max <= EZ_ABSOLUTE_MAX,
    }
    final_theory_fraction = float(
        abs(series.negative_theory[-1] / series.negative_harmonic[0]))
    six_sigma_speed = float(
        THERMAL_SIGMA_MULTIPLIER * run.negative.thermal_velocity)

    overall = (initial_check and negative["pass"] and positive["pass"]
               and all(neutral_checks.values())
               and all(electric_checks.values()))
    return {
        "schema_version": 1,
        "model": "neutral_free_streaming",
        "formula": "a1(t)=a1(0)*exp(-0.5*(k*vT*t)^2)",
        "run": str(run.directory),
        "config": str(run.config_path),
        "frames": run.expected_last_frame + 1,
        "last_time": float(series.time[-1]),
        "theory": {
            "mode_z": run.mode,
            "k": run.k,
            "mec2_kev": MEC2_KEV,
        },
        "criteria": {
            "ballistic_relative_l2_max": BALLISTIC_RELATIVE_L2_MAX,
            "ballistic_max_error_over_a0_max":
                BALLISTIC_MAX_ERROR_OVER_A0_MAX,
            "initial_amplitude_relative_error_max":
                INITIAL_AMPLITUDE_RELATIVE_ERROR_MAX,
            "density_equality_relative_max": DENSITY_EQUALITY_RELATIVE_MAX,
            "charge_relative_max": CHARGE_RELATIVE_MAX,
            "ez_absolute_max": EZ_ABSOLUTE_MAX,
            "minimum_diagnostic_frames": MIN_DIAGNOSTIC_FRAMES,
            "final_theory_fraction_max": FINAL_THEORY_FRACTION_MAX,
            "thermal_sigma_multiplier": THERMAL_SIGMA_MULTIPLIER,
            "thermal_speed_limit": THERMAL_SPEED_LIMIT,
        },
        "coverage": {
            "diagnostic_frames": int(series.time.size),
            "final_theory_fraction": final_theory_fraction,
            "six_sigma_parallel_speed": six_sigma_speed,
        },
        "initial_amplitude": {
            "requested": run.requested_amplitude,
            "cic_factor": cic_factor,
            "expected_deposited": expected_initial,
            "measured": measured_initial,
            "relative_error": initial_amplitude_error,
            "pass": initial_check,
        },
        "species": [negative, positive],
        "neutrality": {
            "max_density_relative_difference": density_difference,
            "max_first_harmonic_difference_over_a0": mode_difference_relative,
            "max_charge_rms_relative": charge_rms,
            "max_charge_absolute_relative": charge_max,
            "checks": neutral_checks,
            "pass": all(neutral_checks.values()),
        },
        "electric_field": {
            "max_ez_rms": electric_rms,
            "max_abs_ez": electric_max,
            "checks": electric_checks,
            "pass": all(electric_checks.values()),
        },
        "pass": overall,
    }


def _write_csv(path: Path, series: SeriesData) -> None:
    fields = [
        "time", "negative_a1_real", "negative_a1_imag", "negative_a1_abs",
        "negative_theory_real", "negative_theory_imag", "negative_theory_abs",
        "positive_a1_real", "positive_a1_imag", "positive_a1_abs",
        "positive_theory_real", "positive_theory_imag", "positive_theory_abs",
        "density_difference_max", "charge_rms_relative",
        "charge_max_relative", "ez_rms", "ez_max",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, time in enumerate(series.time):
            negative = series.negative_harmonic[index]
            negative_theory = series.negative_theory[index]
            positive = series.positive_harmonic[index]
            positive_theory = series.positive_theory[index]
            writer.writerow({
                "time": time,
                "negative_a1_real": negative.real,
                "negative_a1_imag": negative.imag,
                "negative_a1_abs": abs(negative),
                "negative_theory_real": negative_theory.real,
                "negative_theory_imag": negative_theory.imag,
                "negative_theory_abs": abs(negative_theory),
                "positive_a1_real": positive.real,
                "positive_a1_imag": positive.imag,
                "positive_a1_abs": abs(positive),
                "positive_theory_real": positive_theory.real,
                "positive_theory_imag": positive_theory.imag,
                "positive_theory_abs": abs(positive_theory),
                "density_difference_max": series.density_difference_max[index],
                "charge_rms_relative": series.charge_rms_relative[index],
                "charge_max_relative": series.charge_max_relative[index],
                "ez_rms": series.electric_rms[index],
                "ez_max": series.electric_max[index],
            })


def _plot(path: Path, run: RunConfig, series: SeriesData) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(9.5, 10.5), sharex=True)
    colors = ("tab:blue", "tab:orange")
    rows = (
        (run.negative, series.negative_harmonic, series.negative_theory),
        (run.positive, series.positive_harmonic, series.positive_theory),
    )
    for color, (species, measured, theory) in zip(colors, rows):
        rotation = np.conj(measured[0] / abs(measured[0]))
        axes[0].plot(series.time, np.real(measured * rotation), ".",
                     color=color, markersize=4, label=f"{species.name} PIC")
        axes[0].plot(series.time, np.real(theory * rotation), "-",
                     color=color, linewidth=1.7,
                     label=f"{species.name} Maxwellian free streaming")
        axes[1].semilogy(
            series.time,
            np.maximum(np.abs(measured - theory) / abs(measured[0]), 1.0e-16),
            color=color, linewidth=1.4, label=species.name)

    axes[0].set_ylabel(r"rotated complex mode $a_1$")
    axes[0].set_title("Neutral Maxwellian free streaming")
    axes[1].axhline(BALLISTIC_MAX_ERROR_OVER_A0_MAX, color="0.3",
                    linestyle="--", linewidth=1.0, label="acceptance limit")
    axes[1].set_ylabel(r"$|a_1-a_{1,th}|/|a_1(0)|$")

    axes[2].semilogy(
        series.time, np.maximum(series.density_difference_max, 1.0e-18),
        color="tab:green", label="species density difference")
    axes[2].semilogy(
        series.time, np.maximum(series.charge_max_relative, 1.0e-18),
        color="tab:red", label="relative charge residual")
    axes[2].semilogy(
        series.time, np.maximum(series.electric_max, 1.0e-18),
        color="tab:purple", label=r"max $|E_z|$")
    axes[2].set_xlabel(r"$t\,[\omega_{pe}^{-1}]$")
    axes[2].set_ylabel("neutrality / field residual")

    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", required=True,
        help="run directory containing config.json, or config.json itself")
    parser.add_argument(
        "--out-dir", required=True,
        help="directory for free_streaming_results.json/.csv and PNG")
    parser.add_argument(
        "--check", action="store_true",
        help="exit nonzero unless every analytic and neutrality check passes")
    return parser


def run_analysis(args: argparse.Namespace) -> int:
    run = load_config(str(args.run))
    series = load_series(run)
    report = analyse(run, series)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "free_streaming_results.json"
    csv_path = out_dir / "free_streaming_results.csv"
    png_path = out_dir / "free_streaming.png"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    _write_csv(csv_path, series)
    _plot(png_path, run, series)

    print("Neutral free-streaming analysis")
    print(f"  k={run.k:.9e}, vT={run.negative.thermal_velocity:.9e}, "
          f"frames={series.time.size}")
    for result in report["species"]:
        print(f"  {result['name']}: relative L2={result['relative_l2_error']:.4g}, "
              f"max error/a0={result['max_error_over_a0']:.4g}, "
              f"{'PASS' if result['pass'] else 'FAIL'}")
    print(f"  max density mismatch={report['neutrality']['max_density_relative_difference']:.4g}")
    print(f"  max relative charge={report['neutrality']['max_charge_absolute_relative']:.4g}")
    print(f"  max |Ez|={report['electric_field']['max_abs_ez']:.4g}")
    print(f"  overall: {'PASS' if report['pass'] else 'FAIL'}")
    print(f"  wrote {json_path}")
    print(f"  wrote {csv_path}")
    print(f"  wrote {png_path}")
    return 0 if (report["pass"] or not args.check) else 1


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run_analysis(args)
    except FreeStreamingError as exc:
        print(f"free_streaming: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
