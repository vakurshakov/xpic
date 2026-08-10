#!/usr/bin/env python3
"""Analyse ion-sound density-ringdown validation runs.

The observable is the *complex* Fourier coefficient of the ion density,

    a_1(t) = 2 <(n_i/n_i0 - 1) exp(-i k z)>,

and the raw complex samples are fitted to both ion-acoustic branches.  In
particular, no absolute value of a sine/cosine quadrature is fitted: a standing
wave is allowed to change sign and pass through zero.

All physical and numerical parameters come from each run's ``config.json``.
The script deliberately rejects incomplete, non-contiguous, or stale output
trees instead of guessing which frames belong to a run.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

from ion_sound import (
    MEC2_KEV,
    Species,
    acoustic_initial_guess,
    epsilon,
    fit_two_branch,
    solve_dispersion,
    solve_vlasov_poisson,
)


FIT_OMEGA_ERROR_MAX = 0.02
FIT_GAMMA_ERROR_MAX = 0.10
FIT_RESIDUAL_MAX = 0.10
END_ENVELOPE_SNR_MIN = 5.0
STABILITY_OMEGA_SPREAD_MAX = 0.01
STABILITY_GAMMA_SPREAD_MAX = 0.10
CROSS_OMEGA_SPREAD_MAX = 0.01
CROSS_GAMMA_SPREAD_MAX = 0.10
IVP_ROOT_ERROR_MAX = 0.0025  # approximately 0.2%; permits time-grid variation
IVP_RESIDUAL_MAX = 0.02
EXPECTED_PARTICLE_COUNTS = (1024, 2048)
PUBLICATION_GRID_POINTS_Z = 32
LIGHT_PARTICLES_PER_CELL = 512
LIGHT_GRID_POINTS_Z = (32, 64)


@dataclass(frozen=True)
class AnalysisSuite:
    """Numerical comparison and time-window policy for a validation suite."""

    name: str
    comparison: str
    duration_periods: float
    default_fit_window: tuple[float, float]
    stability_starts: tuple[float, ...]
    stability_ends: tuple[float, ...]
    ivp_root_error_max: float


SUITES = {
    "publication": AnalysisSuite(
        name="publication",
        comparison="particles",
        duration_periods=5.0,
        default_fit_window=(0.5, 4.0),
        stability_starts=(0.4, 0.5, 0.6),
        stability_ends=(3.5, 4.0, 4.5),
        ivp_root_error_max=IVP_ROOT_ERROR_MAX,
    ),
    "light": AnalysisSuite(
        name="light",
        comparison="grid",
        duration_periods=3.0,
        default_fit_window=(0.5, 2.5),
        stability_starts=(0.4, 0.5, 0.6),
        stability_ends=(2.2, 2.5, 2.8),
        # A three-period IVP retains a larger non-modal transient than the
        # five-period publication fit.  This remains far tighter than the PIC
        # acceptance thresholds and only validates the chosen short window.
        ivp_root_error_max=0.01,
    ),
}


class RingdownError(RuntimeError):
    """An input tree cannot be analysed without making an assumption."""


@dataclass(frozen=True)
class SpeciesConfig:
    name: str
    n: float
    q: float
    m: float
    temperature_kev: float
    particles_per_cell: int

    def theory(self) -> Species:
        return Species(self.name, self.n, self.q, self.m,
                       self.temperature_kev / MEC2_KEV)


@dataclass
class RunData:
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
    electron: SpeciesConfig
    ion: SpeciesConfig
    mode: int
    amplitude: float
    phase: float
    electron_density_dir: Path
    electron_frame_names: dict[int, str]
    density_dir: Path
    frame_names: dict[int, str]
    expected_last_frame: int
    time: np.ndarray | None = None
    harmonic: np.ndarray | None = None
    noise: np.ndarray | None = None

    @property
    def particles_per_cell(self) -> int:
        counts = {self.electron.particles_per_cell,
                  self.ion.particles_per_cell}
        if len(counts) != 1:
            raise RingdownError(
                f"{self.config_path}: electrons and ions must have the same Np")
        return counts.pop()

    @property
    def k(self) -> float:
        return 2.0 * math.pi * self.mode / self.lz


def _require_mapping(parent: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise RingdownError(f"{where}: required object '{key}' is missing")
    return value


def _number(parent: dict[str, Any], key: str, where: str,
            *, positive: bool = False, nonnegative: bool = False) -> float:
    if key not in parent:
        raise RingdownError(f"{where}: required number '{key}' is missing")
    value = parent[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RingdownError(f"{where}.{key}: expected a JSON number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise RingdownError(f"{where}.{key}: value must be finite")
    if positive and result <= 0.0:
        raise RingdownError(f"{where}.{key}: value must be positive")
    if nonnegative and result < 0.0:
        raise RingdownError(f"{where}.{key}: value must be non-negative")
    return result


def _integer(value: float, where: str, *, positive: bool = False) -> int:
    result = int(round(value))
    if not math.isclose(value, result, rel_tol=0.0, abs_tol=1.0e-9):
        raise RingdownError(f"{where}: expected an integer, got {value!r}")
    if positive and result <= 0:
        raise RingdownError(f"{where}: value must be positive")
    return result


def _vector3(parent: dict[str, Any], key: str, where: str) -> tuple[float, float, float]:
    if key not in parent:
        raise RingdownError(f"{where}: required vector '{key}' is missing")
    values = parent[key]
    if not isinstance(values, list) or len(values) != 3:
        raise RingdownError(f"{where}.{key}: expected a three-element array")
    result = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RingdownError(
                f"{where}.{key}[{index}]: expected a JSON number")
        value = float(value)
        if not math.isfinite(value):
            raise RingdownError(f"{where}.{key}[{index}]: value must be finite")
        result.append(value)
    return tuple(result)  # type: ignore[return-value]


def _close(a: float, b: float, *, scale: float = 1.0) -> bool:
    return math.isclose(a, b, rel_tol=1.0e-11,
                        abs_tol=1.0e-12 * max(scale, 1.0))


def resolve_run(argument: str) -> tuple[Path, Path]:
    """Resolve either a run directory, config path, or output run name."""
    supplied = Path(argument).expanduser()
    candidates: list[Path] = [supplied]
    if not supplied.is_absolute():
        candidates.append(Path(__file__).resolve().parent / "output" / supplied)

    for candidate in candidates:
        if candidate.is_file() and candidate.name == "config.json":
            return candidate.parent.resolve(), candidate.resolve()
        if candidate.is_dir() and (candidate / "config.json").is_file():
            directory = candidate.resolve()
            return directory, directory / "config.json"
    searched = ", ".join(str(path) for path in candidates)
    raise RingdownError(
        f"run '{argument}' was not found; expected config.json in one of: {searched}")


def _parse_species(config: dict[str, Any], config_path: Path) \
        -> tuple[SpeciesConfig, SpeciesConfig]:
    particles = config.get("Particles")
    if not isinstance(particles, list) or len(particles) != 2:
        raise RingdownError(
            f"{config_path}: exactly two particle species are required")

    parsed: list[SpeciesConfig] = []
    for index, item in enumerate(particles):
        where = f"{config_path}:Particles[{index}]"
        if not isinstance(item, dict):
            raise RingdownError(f"{where}: expected an object")
        name = item.get("sort_name")
        if not isinstance(name, str) or not name:
            raise RingdownError(f"{where}.sort_name: required non-empty string")
        # Ringdown is one-dimensional along B.  Tz is deliberately required:
        # accepting the older isotropic T key would silently analyse a
        # different equilibrium than MaxwellianVelocityQuiet loads here.
        temperature = _number(item, "Tz", where, positive=True)
        tx = _number(item, "Tx", where, nonnegative=True)
        ty = _number(item, "Ty", where, nonnegative=True)
        if not _close(tx, 0.0) or not _close(ty, 0.0):
            raise RingdownError(
                f"{where}: canonical ringdown requires Tx=Ty=0")
        if item.get("coord_is_gc") is not True:
            raise RingdownError(
                f"{where}: canonical ringdown requires coord_is_gc=true")
        np_cell = _integer(_number(item, "Np", where, positive=True),
                           f"{where}.Np", positive=True)
        parsed.append(SpeciesConfig(
            name=name,
            n=_number(item, "n", where, positive=True),
            q=_number(item, "q", where),
            m=_number(item, "m", where, positive=True),
            temperature_kev=temperature,
            particles_per_cell=np_cell,
        ))

    electrons = [species for species in parsed if species.q < 0.0]
    ions = [species for species in parsed if species.q > 0.0]
    if len(electrons) != 1 or len(ions) != 1:
        raise RingdownError(
            f"{config_path}: require exactly one q<0 electron and one q>0 ion")
    return electrons[0], ions[0]


def _require_strict_push(config: dict[str, Any], config_path: Path) -> None:
    """Require terminal adaptive-push failures to be fatal for validation."""
    push = config.get("DriftKineticPush")
    key = "fail_on_terminal_nonconvergence"
    if not isinstance(push, dict) or push.get(key) is not True:
        raise RingdownError(
            f"{config_path}: DriftKineticPush.{key}=true is required")


def _particle_initial_conditions(config: dict[str, Any], species_names: Iterable[str],
                                 config_path: Path) -> tuple[int, float, float]:
    """Return shared (mode, amplitude, phase), validating paired quiet loading."""
    species_names = tuple(species_names)
    species_set = set(species_names)
    presets = config.get("Presets")
    if not isinstance(presets, list):
        raise RingdownError(f"{config_path}: required Presets array is missing")
    paired_candidates: list[tuple[int, dict[str, Any]]] = []
    for index, preset in enumerate(presets):
        if not isinstance(preset, dict) or preset.get("command") != "SetParticles":
            continue
        owner = preset.get("particles")
        if not isinstance(owner, str):
            raise RingdownError(
                f"{config_path}:Presets[{index}].particles is required")
        paired = preset.get("paired_with")
        if paired is not None and not isinstance(paired, str):
            raise RingdownError(
                f"{config_path}:Presets[{index}].paired_with must be a string")
        if {owner, paired} == species_set:
            paired_candidates.append((index, preset))
        elif owner in species_set or paired in species_set:
            raise RingdownError(
                f"{config_path}: electrons and ions must be connected by one "
                "shared paired SetParticles preset")
    if len(paired_candidates) != 1:
        raise RingdownError(
            f"{config_path}: require exactly one paired SetParticles preset "
            "connecting the electron and ion species")

    pair_index, pair = paired_candidates[0]
    where = f"{config_path}:Presets[{pair_index}]"
    coordinate = _require_mapping(pair, "coordinate", where)
    momentum = _require_mapping(pair, "momentum", where)
    paired_momentum = _require_mapping(pair, "momentum_paired", where)
    if coordinate.get("name") != "CoordinateInBoxQuietSineExactPaired":
        raise RingdownError(
            f"{where}: require CoordinateInBoxQuietSineExactPaired")
    for label, block in (("momentum", momentum),
                         ("momentum_paired", paired_momentum)):
        if block.get("name") != "MaxwellianVelocityQuiet":
            raise RingdownError(
                f"{where}.{label}: require MaxwellianVelocityQuiet")
    amplitude = _vector3(coordinate, "amplitude", f"{where}.coordinate")
    wave_number = _vector3(coordinate, "wave_number", f"{where}.coordinate")
    phase = _vector3(coordinate, "phase", f"{where}.coordinate")
    if any(not _close(value, 0.0)
           for value in (*amplitude[:2], *wave_number[:2])):
        raise RingdownError(f"{config_path}: ringdown perturbation must be z-only")
    mode = _integer(wave_number[2], f"{where}.coordinate.wave_number[2]",
                    positive=True)
    if amplitude[2] <= 0.0:
        raise RingdownError(f"{config_path}: density amplitude must be positive")
    reference = (mode, amplitude[2], phase[2])

    # E=0 is part of the test, so require it explicitly rather than relying on
    # the solver's default field allocation.
    zero_electric = False
    for index, preset in enumerate(presets):
        if not isinstance(preset, dict) or preset.get("command") != "SetElectricField":
            continue
        if preset.get("field") != "E":
            continue
        setter = _require_mapping(
            preset, "setter", f"{config_path}:Presets[{index}]")
        if setter.get("name") != "SetUniformField":
            raise RingdownError(
                f"{config_path}: initial E must use SetUniformField")
        value = _vector3(setter, "value", f"{config_path}:Presets[{index}].setter")
        zero_electric = all(_close(component, 0.0) for component in value)
    if not zero_electric:
        raise RingdownError(f"{config_path}: explicit E(t=0)=0 preset is required")
    return reference


def _density_diagnostic(config: dict[str, Any], species: str,
                        run_dir: Path, geometry_period: float,
                        config_path: Path) -> tuple[Path, float]:
    diagnostics = config.get("Diagnostics")
    if not isinstance(diagnostics, list):
        raise RingdownError(f"{config_path}: required Diagnostics array is missing")
    matches = [item for item in diagnostics
               if isinstance(item, dict)
               and item.get("diagnostic") == "DistributionMoment"
               and item.get("particles") == species
               and item.get("moment") == "density"]
    if len(matches) != 1:
        raise RingdownError(
            f"{config_path}: require exactly one density diagnostic for '{species}'")
    diagnostic = matches[0]
    out_dir = diagnostic.get("out_dir")
    if not isinstance(out_dir, str) or not out_dir:
        raise RingdownError(
            f"{config_path}: density diagnostic for '{species}' needs explicit out_dir")
    if Path(out_dir).is_absolute():
        raise RingdownError(
            f"{config_path}: density out_dir must be relative to the run directory")
    if "diagnose_period" in diagnostic:
        diagnostic_period = _number(
            diagnostic, "diagnose_period", f"{config_path}:{species}.density",
            positive=True)
        if not _close(diagnostic_period, geometry_period,
                      scale=geometry_period):
            raise RingdownError(
                f"{config_path}: per-diagnostic period is unsupported unless it "
                "equals Geometry.diagnose_period")
    return run_dir / out_dir, geometry_period


def _electric_diagnostic(config: dict[str, Any], run_dir: Path,
                         geometry_period: float, config_path: Path) -> Path:
    """Require the sole field diagnostic to contain only E_z."""
    diagnostics = config.get("Diagnostics")
    if not isinstance(diagnostics, list):
        raise RingdownError(f"{config_path}: required Diagnostics array is missing")
    matches = [item for item in diagnostics
               if isinstance(item, dict)
               and item.get("diagnostic") == "FieldView"
               and item.get("field") == "E"]
    if len(matches) != 1:
        raise RingdownError(
            f"{config_path}: require exactly one FieldView diagnostic for E_z")
    diagnostic = matches[0]
    if diagnostic.get("component") != "z":
        raise RingdownError(
            f"{config_path}: E FieldView must explicitly select component='z'")
    out_dir = diagnostic.get("out_dir")
    if not isinstance(out_dir, str) or not out_dir:
        raise RingdownError(f"{config_path}: E_z diagnostic needs explicit out_dir")
    if Path(out_dir).is_absolute():
        raise RingdownError(
            f"{config_path}: E_z out_dir must be relative to the run directory")
    if "diagnose_period" in diagnostic:
        diagnostic_period = _number(
            diagnostic, "diagnose_period", f"{config_path}:E_z", positive=True)
        if not _close(diagnostic_period, geometry_period,
                      scale=geometry_period):
            raise RingdownError(
                f"{config_path}: E_z diagnose_period must equal "
                "Geometry.diagnose_period")
    return run_dir / out_dir


def _numeric_frames(directory: Path, expected_bytes: int, config_path: Path,
                    expected_last: int) -> dict[int, str]:
    if not directory.is_dir():
        raise RingdownError(f"{config_path}: density directory not found: {directory}")
    result: dict[int, str] = {}
    for entry in directory.iterdir():
        if not entry.is_file() or not entry.name.isdigit():
            continue
        index = int(entry.name)
        if index in result:
            raise RingdownError(
                f"{directory}: duplicate numeric frame index {index}")
        size = entry.stat().st_size
        if size != expected_bytes:
            raise RingdownError(
                f"{entry}: expected {expected_bytes} bytes, found {size}")
        result[index] = entry.name

    expected = set(range(expected_last + 1))
    actual = set(result)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            preview = ", ".join(str(value) for value in missing[:8])
            details.append(f"missing frame(s) {preview}" +
                           (" ..." if len(missing) > 8 else ""))
        if extra:
            preview = ", ".join(str(value) for value in extra[:8])
            details.append(f"stale extra frame(s) {preview}" +
                           (" ..." if len(extra) > 8 else ""))
        raise RingdownError(f"{directory}: " + "; ".join(details))
    return result


def load_run(argument: str) -> RunData:
    run_dir, config_path = resolve_run(argument)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RingdownError(f"cannot read {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise RingdownError(f"{config_path}: root JSON value must be an object")

    _require_strict_push(config, config_path)

    geometry = _require_mapping(config, "Geometry", str(config_path))
    lx = _number(geometry, "x", f"{config_path}:Geometry", positive=True)
    ly = _number(geometry, "y", f"{config_path}:Geometry", positive=True)
    lz = _number(geometry, "z", f"{config_path}:Geometry", positive=True)
    duration = _number(geometry, "t", f"{config_path}:Geometry", positive=True)
    dx = _number(geometry, "dx", f"{config_path}:Geometry", positive=True)
    dy = _number(geometry, "dy", f"{config_path}:Geometry", positive=True)
    dz = _number(geometry, "dz", f"{config_path}:Geometry", positive=True)
    dt = _number(geometry, "dt", f"{config_path}:Geometry", positive=True)
    diagnose_period = _number(
        geometry, "diagnose_period", f"{config_path}:Geometry", positive=True)
    nx = _integer(lx / dx, f"{config_path}:Geometry.x/dx", positive=True)
    ny = _integer(ly / dy, f"{config_path}:Geometry.y/dy", positive=True)
    nz = _integer(lz / dz, f"{config_path}:Geometry.z/dz", positive=True)
    nt = _integer(duration / dt, f"{config_path}:Geometry.t/dt", positive=True)
    diagnostic_stride = _integer(
        diagnose_period / dt,
        f"{config_path}:Geometry.diagnose_period/dt", positive=True)
    expected_last = nt // diagnostic_stride

    electron, ion = _parse_species(config, config_path)
    equilibrium_charge = electron.q * electron.n + ion.q * ion.n
    charge_scale = abs(electron.q * electron.n) + abs(ion.q * ion.n)
    if abs(equilibrium_charge) > 1.0e-12 * max(charge_scale, 1.0):
        raise RingdownError(
            f"{config_path}: equilibrium is not charge neutral "
            f"(sum(q_s*n_s)={equilibrium_charge:.6e})")
    mode, amplitude, phase = _particle_initial_conditions(
        config, (electron.name, ion.name), config_path)
    electron_density_dir, _ = _density_diagnostic(
        config, electron.name, run_dir, diagnose_period, config_path)
    density_dir, diagnose_period = _density_diagnostic(
        config, ion.name, run_dir, diagnose_period, config_path)
    diagnostics = config.get("Diagnostics")
    if not isinstance(diagnostics, list) or len(diagnostics) != 3:
        raise RingdownError(
            f"{config_path}: save only E_z and the two species densities")
    electric_dir = _electric_diagnostic(
        config, run_dir, diagnose_period, config_path)
    expected_bytes = nx * ny * nz * 4
    electron_frames = _numeric_frames(
        electron_density_dir, expected_bytes, config_path, expected_last)
    frames = _numeric_frames(
        density_dir, expected_bytes, config_path, expected_last)
    _numeric_frames(electric_dir, expected_bytes, config_path, expected_last)

    return RunData(
        argument=argument, directory=run_dir, config_path=config_path,
        config=config, nx=nx, ny=ny, nz=nz, lx=lx, ly=ly, lz=lz,
        dx=dx, dy=dy, dz=dz, dt=dt, duration=duration,
        diagnose_period=diagnose_period, electron=electron, ion=ion,
        mode=mode, amplitude=amplitude, phase=phase,
        electron_density_dir=electron_density_dir,
        electron_frame_names=electron_frames,
        density_dir=density_dir, frame_names=frames,
        expected_last_frame=expected_last,
    )


def _comparison_config(config: dict[str, Any],
                       suite: AnalysisSuite) -> dict[str, Any]:
    """Return config with the suite's allowed cross-run differences removed."""
    result = copy.deepcopy(config)
    result.pop("OutputDirectory", None)
    if suite.comparison == "particles":
        particles = result.get("Particles")
        if isinstance(particles, list):
            for item in particles:
                if isinstance(item, dict):
                    item.pop("Np", None)
    elif suite.comparison == "grid":
        geometry = result.get("Geometry")
        if isinstance(geometry, dict):
            geometry.pop("dz", None)
    else:  # pragma: no cover - AnalysisSuite is defined locally and statically
        raise AssertionError(f"unknown comparison kind {suite.comparison!r}")
    return result


def _first_difference(left: Any, right: Any, path: str = "config") -> str | None:
    if type(left) is not type(right):
        return f"{path}: {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if set(left) != set(right):
            only_left = sorted(set(left) - set(right))
            only_right = sorted(set(right) - set(left))
            return f"{path}: keys differ (left-only={only_left}, right-only={only_right})"
        for key in sorted(left):
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: lengths {len(left)} != {len(right)}"
        for index, (a, b) in enumerate(zip(left, right)):
            difference = _first_difference(a, b, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


def validate_matching_configs(runs: list[RunData], suite: AnalysisSuite) -> None:
    if not runs:
        raise RingdownError("at least one run is required")
    if suite.comparison == "particles":
        invalid = [run for run in runs if run.nz != PUBLICATION_GRID_POINTS_Z]
        if invalid:
            details = ", ".join(
                f"{run.directory.name}:Nz={run.nz}" for run in invalid)
            raise RingdownError(
                f"{suite.name} ringdown requires Nz="
                f"{PUBLICATION_GRID_POINTS_Z}; found {details}")
    elif suite.comparison == "grid":
        allowed = set(LIGHT_GRID_POINTS_Z)
        invalid = [run for run in runs if run.nz not in allowed]
        if invalid:
            details = ", ".join(
                f"{run.directory.name}:Nz={run.nz}" for run in invalid)
            expected = "/".join(str(value) for value in LIGHT_GRID_POINTS_Z)
            raise RingdownError(
                f"{suite.name} ringdown allows only Nz={expected}; "
                f"found {details}")
    reference = _comparison_config(runs[0].config, suite)
    for run in runs[1:]:
        difference = _first_difference(
            reference, _comparison_config(run.config, suite))
        if difference:
            allowed = ("Particles[*].Np and OutputDirectory"
                       if suite.comparison == "particles"
                       else "Geometry.dz and OutputDirectory")
            raise RingdownError(
                f"{suite.name} ringdown configs may differ only in {allowed}; "
                f"{run.config_path}: {difference}")


def load_complex_harmonic(run: RunData) -> None:
    z = (np.arange(run.nz, dtype=float) + 0.5) * run.dz
    kernel = np.exp(-1j * run.k * z)
    excluded = {run.mode, 2 * run.mode}
    # Use low unresolved modes only.  CIC attenuation suppresses the high-m
    # tail near Nyquist, so including that tail would understate the particle
    # floor that contaminates the signal mode and overstate the SNR.
    noise_modes = [index for index in range(1, min(15, run.nz // 2) + 1)
                   if index not in excluded]
    if not noise_modes:
        raise RingdownError(
            f"{run.config_path}: grid has no unexcited low modes for noise estimate")

    times: list[float] = []
    harmonics: list[complex] = []
    noise: list[float] = []
    cells = run.nx * run.ny * run.nz
    for index in range(run.expected_last_frame + 1):
        path = run.density_dir / run.frame_names[index]
        data = np.fromfile(path, dtype=np.float32, count=cells)
        if data.size != cells:  # size was checked earlier; catches a concurrent write
            raise RingdownError(f"{path}: frame changed while it was being read")
        if not np.all(np.isfinite(data)):
            raise RingdownError(f"{path}: density frame contains a non-finite value")
        profile = data.reshape(run.nz, run.ny, run.nx).mean(axis=(1, 2))
        relative = profile / run.ion.n - 1.0
        harmonics.append(complex(2.0 * np.mean(relative * kernel)))
        spectrum = 2.0 * np.abs(np.fft.rfft(relative)) / run.nz
        noise.append(float(np.median(spectrum[noise_modes]))
                     if noise_modes else math.nan)
        times.append(index * run.diagnose_period)
    run.time = np.asarray(times, dtype=float)
    run.harmonic = np.asarray(harmonics, dtype=complex)
    run.noise = np.asarray(noise, dtype=float)

    # Paired coordinates must make the realized initial charge perturbation
    # vanish, not merely request the same amplitude in JSON.
    electron_path = (run.electron_density_dir
                     / run.electron_frame_names[0])
    electron_data = np.fromfile(electron_path, dtype=np.float32, count=cells)
    if electron_data.size != cells or not np.all(np.isfinite(electron_data)):
        raise RingdownError(
            f"{electron_path}: invalid or non-finite initial density frame")
    ion_path = run.density_dir / run.frame_names[0]
    ion_data = np.fromfile(ion_path, dtype=np.float32, count=cells)
    if ion_data.size != cells or not np.all(np.isfinite(ion_data)):
        raise RingdownError(f"{ion_path}: invalid or non-finite initial density frame")
    charge_density = (run.electron.q * electron_data.astype(float)
                      + run.ion.q * ion_data.astype(float))
    charge_scale = abs(run.electron.q * run.electron.n) \
        + abs(run.ion.q * run.ion.n)
    charge_tolerance = 1.0e-6 * max(charge_scale, 1.0)
    if float(np.max(np.abs(charge_density))) > charge_tolerance:
        raise RingdownError(
            f"{run.config_path}: frame-0 paired charge density is not zero; "
            f"max(abs(rho))={np.max(np.abs(charge_density)):.6e}")
    electron_profile = electron_data.reshape(
        run.nz, run.ny, run.nx).mean(axis=(1, 2))
    electron_relative = electron_profile / run.electron.n - 1.0
    electron_harmonic = complex(
        2.0 * np.mean(electron_relative * kernel))
    tolerance = max(1.0e-8, 1.0e-6 * abs(run.harmonic[0]))
    if abs(electron_harmonic - run.harmonic[0]) > tolerance:
        raise RingdownError(
            f"{run.config_path}: initial electron and ion density harmonics "
            f"differ by {abs(electron_harmonic - run.harmonic[0]):.6e}")


def two_branch_model(time: np.ndarray, reference_time: float, omega: float,
                     gamma: float, a_plus: complex, a_minus: complex) -> np.ndarray:
    tau = np.asarray(time, dtype=float) - reference_time
    return np.exp(-gamma * tau) * (
        a_plus * np.exp(-1j * omega * tau)
        + a_minus * np.exp(+1j * omega * tau))


def fit_complex_ringdown(time: np.ndarray, harmonic: np.ndarray,
                         omega_guess: float, gamma_guess: float,
                         period: float, window: tuple[float, float],
                         *, n_grid: int = 41, n_passes: int = 7) -> dict[str, Any]:
    lower, upper = window
    if not (0.0 <= lower < upper):
        raise RingdownError(f"invalid fit window {window!r}")
    time = np.asarray(time, dtype=float)
    harmonic = np.asarray(harmonic, dtype=complex)
    if time.ndim != 1 or harmonic.ndim != 1 or time.shape != harmonic.shape:
        raise RingdownError("fit time and harmonic must be equal-length 1-D arrays")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(harmonic)):
        raise RingdownError("fit data contain a non-finite value")
    mask = (time >= lower * period - 1.0e-12 * period) \
        & (time <= upper * period + 1.0e-12 * period)
    if np.count_nonzero(mask) < 12:
        raise RingdownError(
            f"fit window [{lower}, {upper}] T contains fewer than 12 frames")
    selected_time = time[mask]
    selected_harmonic = harmonic[mask]
    if float(np.linalg.norm(selected_harmonic)) <= np.finfo(float).tiny:
        raise RingdownError(
            f"fit window [{lower}, {upper}] T contains a zero signal")
    reference_time = float(selected_time[0])
    omega, gamma, a_plus, a_minus, residual = fit_two_branch(
        selected_time - reference_time, selected_harmonic,
        omega_guess, gamma_guess, n_grid=n_grid, n_passes=n_passes)
    fitted_values = (omega, gamma, a_plus.real, a_plus.imag,
                     a_minus.real, a_minus.imag, residual)
    if not all(math.isfinite(float(value)) for value in fitted_values):
        raise RingdownError(
            f"two-branch fit on [{lower}, {upper}] T returned a non-finite value")
    return {
        "window_periods": [lower, upper],
        "reference_time": reference_time,
        "last_time": float(selected_time[-1]),
        "sample_count": int(selected_time.size),
        "omega": float(omega),
        "gamma": float(gamma),
        "a_plus": complex(a_plus),
        "a_minus": complex(a_minus),
        "relative_residual": float(residual),
        "mask": mask,
    }


def _complex_json(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag),
            "abs": float(abs(value))}


def _public_fit(fit: dict[str, Any]) -> dict[str, Any]:
    return {
        "window_periods": fit["window_periods"],
        "reference_time": fit["reference_time"],
        "last_time": fit["last_time"],
        "sample_count": fit["sample_count"],
        "omega": fit["omega"],
        "gamma": fit["gamma"],
        "a_plus": _complex_json(fit["a_plus"]),
        "a_minus": _complex_json(fit["a_minus"]),
        "relative_residual": fit["relative_residual"],
    }


def _theory_from_run(run: RunData) -> dict[str, Any]:
    electron = run.electron.theory()
    ion = run.ion.theory()
    guess_omega, guess_gamma = acoustic_initial_guess(run.k, electron, ion)
    root = solve_dispersion(
        run.k, [electron, ion], guess_omega - 1j * guess_gamma)
    if not (root.real > 0.0 and root.imag < 0.0):
        raise RingdownError(
            f"{run.config_path}: dispersion solver returned non-acoustic root {root}")
    dispersion_residual = abs(epsilon(root, run.k, [electron, ion]))
    if not math.isfinite(dispersion_residual) or dispersion_residual > 1.0e-9:
        raise RingdownError(
            f"{run.config_path}: dispersion root residual is "
            f"{dispersion_residual:.6e}")
    return {
        "electron": electron,
        "ion": ion,
        "species": [electron, ion],
        "root": root,
        "dispersion_residual": float(dispersion_residual),
        "omega": float(root.real),
        "gamma": float(-root.imag),
        "period": float(2.0 * math.pi / root.real),
        "k": run.k,
        "k_lambda_de": run.k * electron.lamD,
    }


def _validate_duration(run: RunData, period: float,
                       suite: AnalysisSuite) -> None:
    expected = suite.duration_periods * period
    if abs(run.duration - expected) > run.dt + 1.0e-10 * expected:
        raise RingdownError(
            f"{run.config_path}: Geometry.t={run.duration:.12g} is not "
            f"{suite.duration_periods:g}T="
            f"{expected:.12g} within one dt={run.dt:.12g}")
    last_diagnostic_time = run.expected_last_frame * run.diagnose_period
    if last_diagnostic_time < expected - run.diagnose_period - 1.0e-10 * expected:
        raise RingdownError(
            f"{run.config_path}: last complete diagnostic is earlier than "
            f"{suite.duration_periods:g}T "
            "by more than one diagnostic period")


def _analyse_ivp(reference: RunData, theory: dict[str, Any],
                 fit_window: tuple[float, float],
                 suite: AnalysisSuite) -> tuple[dict[str, Any], np.ndarray,
                                                np.ndarray, np.ndarray]:
    period = theory["period"]
    species = theory["species"]
    # solve_vlasov_poisson expects the relative coefficient multiplying F0.
    # The corresponding physical density harmonic is -i alpha n_s.
    phase_factor = -1j * reference.amplitude * np.exp(1j * reference.phase)
    density = {item.name: phase_factor for item in species}
    velocity = {item.name: 0.0j for item in species}
    end_period = max(suite.duration_periods, max(suite.stability_ends))
    time, harmonics, _ = solve_vlasov_poisson(
        species, density, velocity, theory["k"], end_period * period,
        n_record=1200)
    ion_signal = harmonics[theory["ion"].name] / theory["ion"].n
    fit = fit_complex_ringdown(
        time, ion_signal, theory["omega"], theory["gamma"], period,
        fit_window)
    omega_error = abs(fit["omega"] / theory["omega"] - 1.0)
    gamma_error = abs(fit["gamma"] / theory["gamma"] - 1.0)
    checks = {
        "omega_error": omega_error <= suite.ivp_root_error_max,
        "gamma_error": gamma_error <= suite.ivp_root_error_max,
        "fit_residual": fit["relative_residual"] <= IVP_RESIDUAL_MAX,
    }
    result = {
        "fit": _public_fit(fit),
        "omega_relative_error": omega_error,
        "gamma_relative_error": gamma_error,
        "checks": checks,
        "pass": all(checks.values()),
    }
    model = two_branch_model(time, fit["reference_time"], fit["omega"],
                             fit["gamma"], fit["a_plus"], fit["a_minus"])
    return result, time, ion_signal, model


def analyse_run(run: RunData, theory: dict[str, Any],
                fit_window: tuple[float, float],
                suite: AnalysisSuite) -> tuple[dict[str, Any], dict[str, Any]]:
    assert run.time is not None and run.harmonic is not None and run.noise is not None
    main = fit_complex_ringdown(
        run.time, run.harmonic, theory["omega"], theory["gamma"],
        theory["period"], fit_window)

    stability_fits = []
    for start in suite.stability_starts:
        for end in suite.stability_ends:
            stability_fits.append(fit_complex_ringdown(
                run.time, run.harmonic, theory["omega"], theory["gamma"],
                theory["period"], (start, end), n_grid=31, n_passes=6))
    omega_values = np.asarray([fit["omega"] for fit in stability_fits])
    gamma_values = np.asarray([fit["gamma"] for fit in stability_fits])
    omega_spread = float(np.ptp(omega_values) / theory["omega"])
    gamma_spread = float(np.ptp(gamma_values) / theory["gamma"])

    fit_mask = main["mask"]
    finite_noise = run.noise[fit_mask]
    finite_noise = finite_noise[np.isfinite(finite_noise) & (finite_noise >= 0.0)]
    if finite_noise.size == 0:
        raise RingdownError(f"{run.config_path}: cannot estimate harmonic noise")
    noise_floor = float(np.median(finite_noise))
    elapsed = main["last_time"] - main["reference_time"]
    end_envelope = float((abs(main["a_plus"]) + abs(main["a_minus"]))
                         * math.exp(-main["gamma"] * elapsed))
    end_snr_for_check = (end_envelope / noise_floor
                         if noise_floor > 0.0 else math.inf)
    # JSON has no representation for infinity.  A null SNR together with a
    # zero noise_floor unambiguously records the noiseless case.
    end_snr = (float(end_snr_for_check)
               if math.isfinite(end_snr_for_check) else None)

    omega_error = abs(main["omega"] / theory["omega"] - 1.0)
    gamma_error = abs(main["gamma"] / theory["gamma"] - 1.0)
    checks = {
        "omega_error": omega_error <= FIT_OMEGA_ERROR_MAX,
        "gamma_error": gamma_error <= FIT_GAMMA_ERROR_MAX,
        "fit_residual": main["relative_residual"] <= FIT_RESIDUAL_MAX,
        "end_envelope_snr": end_snr_for_check >= END_ENVELOPE_SNR_MIN,
        "window_omega_stability": omega_spread <= STABILITY_OMEGA_SPREAD_MAX,
        "window_gamma_stability": gamma_spread <= STABILITY_GAMMA_SPREAD_MAX,
    }
    result = {
        "name": run.directory.name,
        "directory": str(run.directory),
        "config": str(run.config_path),
        "particles_per_cell": run.particles_per_cell,
        "nz": run.nz,
        "dz": run.dz,
        "frames": len(run.frame_names),
        "last_frame": run.expected_last_frame,
        "last_time": float(run.time[-1]),
        "last_time_periods": float(run.time[-1] / theory["period"]),
        "fit": _public_fit(main),
        "omega_relative_error": omega_error,
        "gamma_relative_error": gamma_error,
        "noise_floor": noise_floor,
        "end_branch_envelope": end_envelope,
        "end_envelope_snr": end_snr,
        "window_stability": {
            "omega_relative_spread": omega_spread,
            "gamma_relative_spread": gamma_spread,
            "fits": [_public_fit(fit) for fit in stability_fits],
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    plotting = {"main_fit": main}
    return result, plotting


def _cross_run_result(results: list[dict[str, Any]],
                      suite: AnalysisSuite) -> dict[str, Any]:
    particle_counts = sorted(result["particles_per_cell"] for result in results)
    grid_points_z = sorted(result["nz"] for result in results)
    grid_steps_z = sorted(result["dz"] for result in results)
    if suite.comparison == "particles":
        parameter_name = "particles_per_cell"
        parameter_values = particle_counts
        parameter_check = tuple(particle_counts) == EXPECTED_PARTICLE_COUNTS
        parameter_checks = {"particle_counts": parameter_check}
    elif suite.comparison == "grid":
        parameter_name = "grid_points_z"
        parameter_values = grid_points_z
        common_particle_count = (
            bool(particle_counts)
            and len(set(particle_counts)) == 1
            and particle_counts[0] == LIGHT_PARTICLES_PER_CELL)
        parameter_checks = {
            "grid_points_z": tuple(grid_points_z) == LIGHT_GRID_POINTS_Z,
            "common_particle_count": common_particle_count,
        }
    else:  # pragma: no cover - AnalysisSuite is defined locally and statically
        raise AssertionError(f"unknown comparison kind {suite.comparison!r}")
    if len(results) < 2:
        return {
            "available": False,
            "comparison": suite.comparison,
            "comparison_parameter": parameter_name,
            "comparison_values": parameter_values,
            "particle_counts": particle_counts,
            "grid_points_z": grid_points_z,
            "grid_steps_z": grid_steps_z,
            "checks": {
                **parameter_checks,
                "omega_convergence": False,
                "gamma_convergence": False,
            },
            "pass": False,
        }
    omega = np.asarray([result["fit"]["omega"] for result in results])
    gamma = np.asarray([result["fit"]["gamma"] for result in results])
    omega_reference = float(np.mean(omega))
    gamma_reference = float(np.mean(gamma))
    tiny = np.finfo(float).tiny
    omega_spread = float(np.ptp(omega) / max(abs(omega_reference), tiny))
    gamma_spread = float(np.ptp(gamma) / max(abs(gamma_reference), tiny))
    checks = {
        **parameter_checks,
        "omega_convergence": omega_spread <= CROSS_OMEGA_SPREAD_MAX,
        "gamma_convergence": gamma_spread <= CROSS_GAMMA_SPREAD_MAX,
    }
    return {
        "available": True,
        "comparison": suite.comparison,
        "comparison_parameter": parameter_name,
        "comparison_values": parameter_values,
        "particle_counts": particle_counts,
        "grid_points_z": grid_points_z,
        "grid_steps_z": grid_steps_z,
        "omega_relative_spread": omega_spread,
        "gamma_relative_spread": gamma_spread,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "name", "particles_per_cell", "nz", "dz", "frames", "last_time_periods",
        "omega", "omega_relative_error", "gamma", "gamma_relative_error",
        "relative_residual", "noise_floor", "end_branch_envelope",
        "end_envelope_snr", "window_omega_relative_spread",
        "window_gamma_relative_spread", "pass",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "name": result["name"],
                "particles_per_cell": result["particles_per_cell"],
                "nz": result["nz"],
                "dz": result["dz"],
                "frames": result["frames"],
                "last_time_periods": result["last_time_periods"],
                "omega": result["fit"]["omega"],
                "omega_relative_error": result["omega_relative_error"],
                "gamma": result["fit"]["gamma"],
                "gamma_relative_error": result["gamma_relative_error"],
                "relative_residual": result["fit"]["relative_residual"],
                "noise_floor": result["noise_floor"],
                "end_branch_envelope": result["end_branch_envelope"],
                "end_envelope_snr": result["end_envelope_snr"],
                "window_omega_relative_spread":
                    result["window_stability"]["omega_relative_spread"],
                "window_gamma_relative_spread":
                    result["window_stability"]["gamma_relative_spread"],
                "pass": result["pass"],
            })


def _plot(path: Path, runs: list[RunData], results: list[dict[str, Any]],
          plotting: list[dict[str, Any]], theory: dict[str, Any],
          ivp_plot: tuple[np.ndarray, np.ndarray, np.ndarray],
          suite: AnalysisSuite) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    period = theory["period"]
    fig, (ax_signed, ax_magnitude) = plt.subplots(
        2, 1, figsize=(10.0, 9.0), sharex=True)
    colors = plt.get_cmap("tab10").colors
    for index, (run, result, private) in enumerate(zip(runs, results, plotting)):
        assert run.time is not None and run.harmonic is not None and run.noise is not None
        color = colors[index % len(colors)]
        fit = private["main_fit"]
        model = two_branch_model(
            run.time, fit["reference_time"], fit["omega"], fit["gamma"],
            fit["a_plus"], fit["a_minus"])
        initial = run.harmonic[0]
        rotation = np.conj(initial / abs(initial)) if abs(initial) > 0.0 else 1.0
        signed = np.real(run.harmonic * rotation)
        signed_model = np.real(model * rotation)
        elapsed = run.time - fit["reference_time"]
        envelope = (abs(fit["a_plus"]) + abs(fit["a_minus"])) \
            * np.exp(-fit["gamma"] * elapsed)
        if suite.comparison == "grid":
            label = (f"{run.directory.name} "
                     f"(Nz={run.nz}, dz={run.dz:.4g}, Np={run.particles_per_cell})")
        else:
            label = f"{run.directory.name} (Np={run.particles_per_cell})"
        ax_signed.plot(run.time / period, signed, marker=".", markersize=3,
                       linewidth=0.9, color=color, alpha=0.75, label=label)
        ax_signed.plot(run.time / period, signed_model, linewidth=1.7,
                       color=color)
        ax_signed.plot(run.time / period, envelope, "--", linewidth=1.0,
                       color=color, alpha=0.75)
        ax_signed.plot(run.time / period, -envelope, "--", linewidth=1.0,
                       color=color, alpha=0.75)
        ax_magnitude.semilogy(run.time / period, np.abs(run.harmonic),
                              marker=".", markersize=3, linewidth=0.9,
                              color=color, label=label)
        ax_magnitude.semilogy(run.time / period, np.maximum(run.noise, 1.0e-30),
                              ":", linewidth=1.3, color=color,
                              label=f"{run.directory.name} unexcited modes")

    ivp_time, ivp_signal, ivp_model = ivp_plot
    ivp_rotation = np.conj(ivp_signal[0] / abs(ivp_signal[0]))
    ax_signed.plot(ivp_time / period, np.real(ivp_model * ivp_rotation),
                   color="black", linewidth=1.2, alpha=0.75,
                   label="linear Vlasov-Poisson IVP fit")

    lower, upper = results[0]["fit"]["window_periods"]
    for axis in (ax_signed, ax_magnitude):
        axis.axvspan(lower, upper, color="0.8", alpha=0.2,
                     label="fit window" if axis is ax_signed else None)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    ax_signed.axhline(0.0, color="0.3", linewidth=0.6)
    ax_signed.set_ylabel(r"signed complex mode $a_1$")
    ax_signed.set_title(
        "Ion-sound density ringdown: raw complex-harmonic fit")
    ax_magnitude.set_xlabel(r"$t/T$")
    ax_magnitude.set_ylabel(r"$|a_1|$ and unexcited-mode noise")
    ax_magnitude.set_xlim(0.0, max(float(run.time[-1] / period) for run in runs))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", nargs="+", required=True, metavar="RUN_DIR",
        help="run directories, config.json paths, or names under output/")
    parser.add_argument(
        "--suite", choices=tuple(SUITES), default="publication",
        help="validation policy: publication compares Np at 5T; light compares "
             "Nz=32/64 at 3T (default: publication)")
    parser.add_argument(
        "--fit-window", nargs=2, type=float, default=None,
        metavar=("START_T", "END_T"),
        help="fit interval in ion-sound periods (suite default: publication "
             "0.5 4.0; light 0.5 2.5)")
    parser.add_argument("--out-dir", required=True,
                        help="directory for ringdown_results.json/.csv and PNG")
    parser.add_argument(
        "--check", action="store_true",
        help="exit nonzero unless all per-run and convergence criteria pass")
    return parser


def run_analysis(args: argparse.Namespace) -> int:
    suite = SUITES[args.suite]
    fit_window = (suite.default_fit_window if args.fit_window is None else
                  (float(args.fit_window[0]), float(args.fit_window[1])))
    if not (0.0 <= fit_window[0] < fit_window[1]):
        raise RingdownError("--fit-window requires 0 <= START_T < END_T")
    if fit_window[1] > max(suite.stability_ends):
        raise RingdownError(
            f"--fit-window END_T must not exceed "
            f"{max(suite.stability_ends):g} T for --suite {suite.name}")

    runs = [load_run(argument) for argument in args.runs]
    validate_matching_configs(runs, suite)
    theory = _theory_from_run(runs[0])
    for run in runs:
        _validate_duration(run, theory["period"], suite)
        if not _close(run.k, theory["k"], scale=theory["k"]):
            raise RingdownError(f"{run.config_path}: wave number differs")
        load_complex_harmonic(run)

    ivp_result, ivp_time, ivp_signal, ivp_model = _analyse_ivp(
        runs[0], theory, fit_window, suite)
    results = []
    plotting = []
    for run in runs:
        result, private = analyse_run(run, theory, fit_window, suite)
        results.append(result)
        plotting.append(private)
    cross = _cross_run_result(results, suite)

    criteria = {
        "fit_omega_relative_error_max": FIT_OMEGA_ERROR_MAX,
        "fit_gamma_relative_error_max": FIT_GAMMA_ERROR_MAX,
        "fit_relative_residual_max": FIT_RESIDUAL_MAX,
        "end_branch_envelope_snr_min": END_ENVELOPE_SNR_MIN,
        "window_omega_relative_spread_max": STABILITY_OMEGA_SPREAD_MAX,
        "window_gamma_relative_spread_max": STABILITY_GAMMA_SPREAD_MAX,
        "cross_run_omega_relative_spread_max": CROSS_OMEGA_SPREAD_MAX,
        "cross_run_gamma_relative_spread_max": CROSS_GAMMA_SPREAD_MAX,
        "ivp_root_relative_error_max": suite.ivp_root_error_max,
        "ivp_relative_residual_max": IVP_RESIDUAL_MAX,
    }
    overall_pass = (ivp_result["pass"] and cross["pass"]
                    and all(result["pass"] for result in results))
    report = {
        "schema_version": 1,
        "suite": suite.name,
        "comparison": suite.comparison,
        "expected_duration_periods": suite.duration_periods,
        "stability_start_periods": list(suite.stability_starts),
        "stability_end_periods": list(suite.stability_ends),
        "observable": "a1=2*mean((n_i/n_i0-1)*exp(-i*k*z))",
        "fit_model": "A_plus*exp((-gamma-i*omega)*tau)+"
                     "A_minus*exp((-gamma+i*omega)*tau)",
        "fit_window_periods": list(fit_window),
        "theory": {
            "k": theory["k"],
            "k_lambda_de": theory["k_lambda_de"],
            "omega": theory["omega"],
            "gamma": theory["gamma"],
            "period": theory["period"],
            "dispersion_residual": theory["dispersion_residual"],
        },
        "criteria": criteria,
        "linear_vlasov_poisson_ivp": ivp_result,
        "runs": results,
        "cross_run": cross,
        "pass": overall_pass,
    }

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ringdown_results.json"
    csv_path = out_dir / "ringdown_results.csv"
    png_path = out_dir / "ringdown.png"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    _write_csv(csv_path, results)
    _plot(png_path, runs, results, plotting, theory,
          (ivp_time, ivp_signal, ivp_model), suite)

    print(f"Ion-sound density-ringdown analysis ({suite.name} suite)")
    print(f"  theory: omega={theory['omega']:.9e}, "
          f"Gamma={theory['gamma']:.9e}, T={theory['period']:.9g}")
    for result in results:
        snr = result["end_envelope_snr"]
        snr_text = "infinite" if snr is None else f"{snr:.3g}"
        grid_text = (f", Nz={result['nz']}, dz={result['dz']:.6g}"
                     if suite.comparison == "grid" else "")
        print(f"  {result['name']}: Np={result['particles_per_cell']}"
              f"{grid_text}, "
              f"omega/theory={result['fit']['omega']/theory['omega']:.6f}, "
              f"Gamma/theory={result['fit']['gamma']/theory['gamma']:.6f}, "
              f"residual={result['fit']['relative_residual']:.4f}, "
              f"end-envelope/noise={snr_text}, "
              f"{'PASS' if result['pass'] else 'FAIL'}")
    print(f"  linear IVP: {'PASS' if ivp_result['pass'] else 'FAIL'}")
    print(f"  {suite.comparison} convergence: "
          f"{'PASS' if cross['pass'] else 'FAIL'}")
    print(f"  overall: {'PASS' if overall_pass else 'FAIL'}")
    print(f"  wrote {json_path}")
    print(f"  wrote {csv_path}")
    print(f"  wrote {png_path}")
    return 0 if (overall_pass or not args.check) else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run_analysis(args)
    except RingdownError as exc:
        print(f"ion_sound_ringdown: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
