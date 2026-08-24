#!/usr/bin/env python3
"""Check hard-coded C++ ion-sound modes against the full-Z grid theory."""

import argparse
import math
import pathlib
import re
import unittest

import ion_sound


CONSTANT_RE = re.compile(
    r"^\s*constexpr\s+double\s+([A-Za-z_]\w*)\s*=\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*;",
    re.MULTILINE)

INPUT_NAMES = {
    "theory_mec2_kev", "theory_Te_kev", "theory_Ti_kev",
    "theory_ne", "theory_ni", "theory_qe", "theory_qi",
    "theory_me", "theory_mi", "theory_Lz", "theory_dz",
    "theory_mode", "theory_dn_i",
}
OUTPUT_NAMES = {
    "omega_real", "gamma", "E_force", "E_grid", "field_grid_phase",
    "a_n_e", "phi_n_e", "a_n_i", "phi_n_i",
}


def read_constants(path):
    text = pathlib.Path(path).read_text(encoding="utf-8")
    constants = {name: float(value) for name, value in CONSTANT_RE.findall(text)}
    missing = sorted((INPUT_NAMES | OUTPUT_NAMES) - constants.keys())
    if missing:
        raise ValueError(
            f"{path}: missing literal constexpr double values: " +
            ", ".join(missing))
    return constants


def calculate_mode(c):
    electron = ion_sound.Species(
        "electrons", c["theory_ne"], c["theory_qe"], c["theory_me"],
        c["theory_Te_kev"] / c["theory_mec2_kev"])
    ion = ion_sound.Species(
        "ions", c["theory_ni"], c["theory_qi"], c["theory_mi"],
        c["theory_Ti_kev"] / c["theory_mec2_kev"])
    species = [electron, ion]
    k = 2.0 * math.pi * c["theory_mode"] / c["theory_Lz"]
    half_cell_phase = 0.5 * k * c["theory_dz"]
    sinc = math.sin(half_cell_phase) / half_cell_phase
    shape_s1 = sinc**2
    coupling = shape_s1**2

    omega_seed, gamma_seed = ion_sound.acoustic_initial_guess(
        k, electron, ion)
    omega = ion_sound.solve_dispersion(
        k, species, omega_seed - 1j * gamma_seed, coupling=coupling)
    residual = abs(1.0 + coupling * (
        ion_sound.epsilon(omega, k, species) - 1.0))
    if residual >= 2.0e-8:
        raise AssertionError(f"full-Z root residual is too large: {residual:.6e}")

    electric_force = ion_sound.field_from_density(
        ion, omega, k, c["theory_dn_i"])
    result = {
        "omega_real": omega.real,
        "gamma": -omega.imag,
        "E_force": electric_force,
        "E_grid": electric_force / shape_s1,
        "field_grid_phase": half_cell_phase,
    }
    for sp, suffix in ((electron, "e"), (ion, "i")):
        amplitude, phase = ion_sound.density_sine(
            *ion_sound.density_amplitudes(sp, omega, k, electric_force))
        result[f"a_n_{suffix}"] = amplitude / sp.n
        result[f"phi_n_{suffix}"] = phase
    return result


class IonSoundCppConstantsTest(unittest.TestCase):
    case_paths = []

    def test_cpp_modes_match_grid_theory(self):
        self.assertTrue(self.case_paths, "no C++ ion-sound cases were supplied")
        for path in self.case_paths:
            with self.subTest(case=path):
                configured = read_constants(path)
                calculated = calculate_mode(configured)
                for name in sorted(OUTPUT_NAMES):
                    self.assertTrue(
                        math.isclose(configured[name], calculated[name],
                                     rel_tol=5.0e-10, abs_tol=5.0e-13),
                        f"{path}: stale {name}: configured="
                        f"{configured[name]:.16e}, theory="
                        f"{calculated[name]:.16e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="+", help="C++ mode files to validate")
    args, unittest_args = parser.parse_known_args()
    IonSoundCppConstantsTest.case_paths = args.cases
    unittest.main(argv=[__file__] + unittest_args)


if __name__ == "__main__":
    main()
