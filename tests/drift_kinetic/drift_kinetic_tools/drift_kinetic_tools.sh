#!/usr/bin/env bash
# Dispatch drift-kinetic post-processing scripts for a given test output.
#
# Usage:
#   ./drift_kinetic_tools.sh <test_name> [diag1 diag2 ...]
#
# Known diagnostics:
#   E B J M rotM curlM
#                  -- FieldView, handled by drift_kinetic_fields.py (Cartesian
#                     components F_x, F_y, F_z). rotM reads the simulation's
#                     rotM diagnostic; curlM reads M and computes ∇×M
#                     numerically on the slice (∂/∂n along the slice normal
#                     is dropped).
#   <field>:<axis>[:<idx>]
#                  -- Slice a 3D FieldView along axis x/y/z at the given cell
#                     index (default N/2 when idx is omitted). Example:
#                     "E:z:10", "M:y". Works for any of the field tokens above
#                     and also under the cyl: prefix (e.g. "cyl:B:z:5").
#   cyl:E cyl:B cyl:J cyl:M cyl:rotM cyl:curlM cyl:dB
#                  -- Same as above but in cylindrical components
#                     (F_r, F_phi, F_z) relative to the cylinder axis,
#                     handled by drift_kinetic_fields_cyl.py
#   dB             -- B - B0 perturbation (FieldView of B + Presets B0)
#   dBfft[:N]      -- ln|dB(k_z = 2*pi*N/L_z, t)| of the parallel-z Fourier
#                     harmonic N (default N=1), handled by
#                     drift_kinetic_dB_fft.py. Pass several modes by
#                     repeating the token: "dBfft:1 dBfft:2 dBfft:3".
#   beta           -- Compare particle-pressure beta vs displaced-field beta
#                     ((<|dB|>_z / |B0|)^2), handled by drift_kinetic_beta.py
#   energy         -- Time series of wE, wB (top) and wK per sort (bottom)
#                     from temporal/energy.txt, handled by
#                     drift_kinetic_energy.py
#   energy_compare[:N]
#                  -- Compare energy diagnostics for all sibling output runs
#                     matching the current test's ex-prefix (for example all
#                     drift_kinetic_energy_ei_ex* runs). Writes three figures:
#                     wE, relative wK_electrons/wK_ions changes, and total-energy
#                     change versus omega_pe*t. Optional N limits omega_pe*t
#                     to [0, N].
#   eq[:METHOD]    -- Radial pressure-balance check: p_perp + B^2/2 vs r,
#                     averaged over azimuthal angle around the cylinder axis;
#                     uses plane-Z B and M plus 3D density z-center slice.
#                     METHOD is "dec" (Cartesian binning, default) or
#                     "pol" (polar bilinear-resampling, uniform precision in
#                     r including near r=0). Pass both as separate tokens to
#                     get both flavours. Handled by
#                     drift_kinetic_equilibrium.py
#   force[:METHOD[:START_IDX]]
#                  -- Radial momentum-balance check
#                     R(r) = -dp_perp/dr + (J + curl M)_phi B_z, time-averaged
#                     and shown as the relative residual <R>_t / |<-dp/dr>_t|.
#                     METHOD is dec (default) or pol (same as eq). START_IDX
#                     skips snapshots with idx < START_IDX before averaging
#                     (default 0). Handled by drift_kinetic_force.py.
#   force_y[:Z_MODE[:START_IDX[:tavg[:TAVG_START_IDX]]]]
#                  -- 1D-along-x momentum-balance check on a plane-Y slice:
#                     four panels — R(x), -dp/dx & -J_phi*B_z,
#                     (p_perp + B^2/2) - baseline, p_perp & B0^2/2 - B^2/2.
#                     Z_MODE is avg (default, average over z) or center
#                     (Nz//2 plane only). START_IDX skips snapshots before it
#                     (default 0). Trailing ":tavg" overlays a black-dotted
#                     running mean of R(x) on panel 1; the optional
#                     TAVG_START_IDX (default = START_IDX) is the first idx
#                     that contributes to that mean. Handled by
#                     drift_kinetic_force_y.py.
#   profiles[:METHOD]
#                  -- Mean radial profiles of E_r, J_phi and B_z overlaid on
#                     a single figure (three y-axes), using the plane-Z slice.
#                     Same dec/pol toggle as eq. Handled by
#                     drift_kinetic_profiles.py.
#   ions electrons -- DistributionMoment density slices (X, Y, Z center planes),
#                     handled by drift_kinetic_density.py; passing both
#                     overlays them on the same figure
#   density_z      -- (x, y)-averaged density profile <n>(z) for every species
#                     with a density diagnostic, all curves on one figure;
#                     handled by drift_kinetic_density_z.py
#   compare_density_z:<other_test>[:<t_max_T>]
#                  -- ln|delta n_c(t)| comparison vs another test's run, on a
#                     single panel (no profile panel). Legend labels carry
#                     each run's per-species Np. The other test must live at
#                     ../output/<other_test>/config.json. Repeat the token to
#                     compare against several runs. Optional <t_max_T> bounds
#                     the right plot's x-axis in units of T (e.g.
#                     compare_density_z:drift_kinetic_sound_ex6:2.75 draws
#                     up to 2.75 T). Handled by compare_density_z.py.
#   jspec[:sub][:M]
#                  -- Ballistic-spectrum diagnostic for ion-sound runs. From
#                     the (x,y)-averaged J_z(z,t) of electrons and ions
#                     computes the z-Fourier mode-m time series, plots
#                     ln|J^(m)(t)| for m=1..3, the parametric
#                     Re J^(1) vs Im J^(1) trajectory, and the temporal
#                     |FFT[J_c^(M)(t)](ω)| against v_‖ = ω/(M k) per
#                     species. Sharp narrow peaks in the spectrum (outside
#                     the wave line at v=c_s) are the signature of PIC
#                     phase-space recurrence / ballistic echo; a smooth
#                     Maxwellian-shaped envelope means no filamentation.
#                     Optional ":sub" subtracts the fitted ion-sound wave
#                     from J before the FFT. Optional ":M" overrides the
#                     spectral mode (default M=1). Handled by
#                     drift_kinetic_J_spectrum.py.
#   fp_y[:Z_MODE[:START_IDX[:END_IDX]]]
#                  -- Two-panel plane-Y diagnostic (force-balance and
#                     p_perp / B_0^2/2 - B^2/2) over a user-selected idx
#                     range. Writes per-frame PNGs, an MP4 video and a
#                     final time-mean PNG. Z_MODE is avg (default) or
#                     center; START_IDX (default 0) and END_IDX (default
#                     last available) bound the frames inclusively.
#                     Handled by drift_kinetic_force_pressure_y.py.
#   fp3d_y[:FLAG...][:Z_MODE[:START_IDX[:END_IDX[:Y_IDX]]]]
#                  -- Same two-panel diagnostic as fp_y, but computed from
#                     3D FieldView + DistributionMoment outputs sliced at
#                     y = Y_IDX (default Ny//2): B, J = sum_s <sort>/J,
#                     rotM = sum_s <sort>/rotM, and p_perp = sum_s n_s *
#                     T_perp,s from the density / temperature_perp moments.
#                     Z_MODE / START_IDX / END_IDX as for fp_y. Zero or more
#                     boolean FLAG sub-tokens may lead, in any order:
#                       Pi     -- top-panel p_perp -> flow Pi_{rr} =
#                                 sum_s m_s J_{x,s}^2/(q_s^2 n_s); hoop fold
#                                 -dPi_{rr}/dr + Pi_{phi phi}/r - Pi_{rr}/r.
#                       avgzy  -- average over y as well as z (Y_IDX ignored).
#                       pidiag -- Pi_{phi phi} from the second component of
#                                 the momentum_flux_diag_cyl moment instead
#                                 of the J-flow estimate.
#                       noj    -- do not backward-average J / rotM in time.
#                     Handled by drift_kinetic_force_pressure_3D_y.py.
#   pcyl[:J][:Z_MODE[:IDX]]
#                  -- Per-species cylindrical pressure tensor diagonals
#                     (Pi_rr, Pi_phi phi, Pi_zz) at one snapshot, plotted
#                     vs r after z-reduction (avg/center) and phi-
#                     averaging. By default each panel overlays the
#                     "full" tensor from the momentum_flux_diag_cyl
#                     diagnostic with the bulk-flow part
#                     Pi_{ii}^flow = m J_i^2 / (q^2 n) computed from J
#                     and the density diagnostic; the difference is the
#                     thermal part. The "J" sub-token skips the
#                     momentum_flux_diag_cyl diagnostic and plots only
#                     the J-based estimate for all three components
#                     (same approximation as fp3d_y's p_phiphi, but for
#                     rr/phi-phi/zz). Default IDX is the last common
#                     snapshot. Handled by drift_kinetic_pressure_cyl.py.
#   fp3d_phi[:Z_MODE[:START_IDX[:END_IDX]]]
#                  -- Same as fp3d_y, but the 3D fields are reduced over z
#                     (avg or center) and then averaged in phi about the
#                     cylinder axis (auto-detected); all profiles are
#                     plotted vs r. Vector quantities are projected to
#                     cylindrical (r, phi, z) before binning. Handled by
#                     drift_kinetic_force_pressure_3D_phi.py.
#   fp1d[:START_IDX[:END_IDX]]
#                  -- Same two-panel diagnostic as fp3d_y, but reduced to a
#                     single 1D profile vs x by averaging over BOTH z and y
#                     (planar Cartesian force balance, no cylindrical hoop
#                     term). p_perp = sum_s 0.5*temperature_perp_s; top panel
#                     is -dp_perp/dx (central diff) vs -(J_y + (rot M)_y) B_z
#                     and -(rot M)_y B_z, where B_z is de-staggered onto the
#                     cell node in x and y and J_y / (rot M)_y in y before the
#                     product, then averaged over z and y. START_IDX /
#                     END_IDX bound the frames inclusively. Handled by
#                     drift_kinetic_force_pressure_1D.py.
#
# Example:
#   ./drift_kinetic_tools.sh drift_kinetic_hose_ex1 E B
#   ./drift_kinetic_tools.sh drift_kinetic_hose_ex1 ions electrons
#   ./drift_kinetic_tools.sh drift_kinetic_hose_ex7 dBfft:1 dBfft:2
#   ./drift_kinetic_tools.sh drift_kinetic_hose_ex1           # all defaults

set -eu

if [ "$#" -lt 1 ]; then
    cat >&2 <<EOF
Usage: $0 <test_name> [diagnostics...]
Diagnostics: E B J M rotM curlM dB cyl:{E,B,J,M,rotM,curlM,dB} dBfft[:N] beta energy energy_compare[:N] eq[:dec|:pol] force[:dec|:pol[:START_IDX]] force_y[:avg|:center[:START_IDX[:tavg[:TAVG_START_IDX]]]] fp_y[:avg|:center[:START_IDX[:END_IDX]]] fp3d_y[:Pi|:avgzy|:pidiag|:noj...][:avg|:center[:START_IDX[:END_IDX[:Y_IDX]]]] fp3d_phi[:avg|:center[:START_IDX[:END_IDX]]] fp1d[:START_IDX[:END_IDX]] pcyl[:J][:avg|:center[:IDX]] profiles[:dec|:pol] ions electrons density_z compare_density_z:<other_test>[:<t_max_T>] jspec[:sub][:M]
EOF
    exit 1
fi

TEST_NAME="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${TEST_DIR}/output/${TEST_NAME}"
CONFIG="${OUTPUT_DIR}/config.json"

if [ ! -f "${CONFIG}" ]; then
    echo "Config not found: ${CONFIG}" >&2
    exit 1
fi

if [ "$#" -eq 0 ]; then
    DIAGS=(E B J M dB ions electrons)
else
    DIAGS=("$@")
fi

FIELDS=()
FIELDS_CYL=()
PARTICLES=()
DBFFT_MODES=()
RUN_BETA=0
RUN_ENERGY=0
ENERGY_COMPARE_TIME_MAX=()
EQ_METHODS=()
FORCE_METHODS=()
FORCE_START_IDX=()
FORCE_Y_ZMODE=()
FORCE_Y_START_IDX=()
FORCE_Y_TAVG=()
FORCE_Y_TAVG_START=()
PROFILES_METHODS=()
RUN_DENSITY_Z=0
COMPARE_DENSITY_Z=()
JSPEC_SUB=()
JSPEC_MODE=()
FP_Y_ZMODE=()
FP_Y_START_IDX=()
FP_Y_END_IDX=()
FP3D_Y_ZMODE=()
FP3D_Y_START_IDX=()
FP3D_Y_END_IDX=()
FP3D_Y_Y_IDX=()
FP3D_Y_PI_MODE=()
FP3D_Y_AVG_ZY=()
FP3D_Y_PI_DIAG=()
FP3D_Y_NO_J=()
FP3D_PHI_ZMODE=()
FP3D_PHI_START_IDX=()
FP3D_PHI_END_IDX=()
FP1D_START_IDX=()
FP1D_END_IDX=()
PCYL_ZMODE=()
PCYL_IDX=()
PCYL_MODE=()

# force[:METHOD[:START_IDX]] — split on ':' into up to three parts.
parse_force_token() {
    local token="$1"
    local method="dec"
    local start="0"
    if [ "${token}" != "force" ]; then
        # strip leading "force:"
        local rest="${token#force:}"
        # method[:start]
        if [[ "${rest}" == *:* ]]; then
            method="${rest%%:*}"
            start="${rest#*:}"
        else
            method="${rest}"
        fi
    fi
    case "${method}" in
        dec|pol) ;;
        *) echo "Unknown force method '${method}' in token '${token}', skipping" >&2
           return 1 ;;
    esac
    if ! [[ "${start}" =~ ^[0-9]+$ ]]; then
        echo "Invalid start_idx '${start}' in token '${token}', skipping" >&2
        return 1
    fi
    FORCE_METHODS+=("${method}")
    FORCE_START_IDX+=("${start}")
}

# force_y[:Z_MODE[:START_IDX[:tavg[:TAVG_START_IDX]]]]
# — split on ':' into up to five parts.
parse_force_y_token() {
    local token="$1"
    local zmode="avg"
    local start="0"
    local tavg="0"
    local tavg_start=""
    if [ "${token}" != "force_y" ]; then
        local rest="${token#force_y:}"
        IFS=':' read -r -a parts <<< "${rest}"
        if [ "${#parts[@]}" -ge 1 ] && [ -n "${parts[0]}" ]; then
            zmode="${parts[0]}"
        fi
        if [ "${#parts[@]}" -ge 2 ] && [ -n "${parts[1]}" ]; then
            start="${parts[1]}"
        fi
        if [ "${#parts[@]}" -ge 3 ] && [ -n "${parts[2]}" ]; then
            case "${parts[2]}" in
                tavg) tavg="1" ;;
                *) echo "Unknown force_y flag '${parts[2]}' in token '${token}', expected 'tavg'" >&2
                   return 1 ;;
            esac
        fi
        if [ "${#parts[@]}" -ge 4 ] && [ -n "${parts[3]}" ]; then
            if [ "${tavg}" != "1" ]; then
                echo "force_y token '${token}': TAVG_START_IDX requires 'tavg' flag" >&2
                return 1
            fi
            tavg_start="${parts[3]}"
        fi
    fi
    case "${zmode}" in
        avg|center) ;;
        *) echo "Unknown force_y z-mode '${zmode}' in token '${token}', expected avg|center" >&2
           return 1 ;;
    esac
    if ! [[ "${start}" =~ ^[0-9]+$ ]]; then
        echo "Invalid start_idx '${start}' in token '${token}', skipping" >&2
        return 1
    fi
    if [ -n "${tavg_start}" ] && ! [[ "${tavg_start}" =~ ^[0-9]+$ ]]; then
        echo "Invalid tavg_start_idx '${tavg_start}' in token '${token}', skipping" >&2
        return 1
    fi
    FORCE_Y_ZMODE+=("${zmode}")
    FORCE_Y_START_IDX+=("${start}")
    FORCE_Y_TAVG+=("${tavg}")
    FORCE_Y_TAVG_START+=("${tavg_start}")
}

# fp_y[:Z_MODE[:START_IDX[:END_IDX]]] — split on ':' into up to four parts.
parse_fp_y_token() {
    local token="$1"
    local zmode="avg"
    local start="0"
    local end=""
    if [ "${token}" != "fp_y" ]; then
        local rest="${token#fp_y:}"
        IFS=':' read -r -a parts <<< "${rest}"
        if [ "${#parts[@]}" -ge 1 ] && [ -n "${parts[0]}" ]; then
            zmode="${parts[0]}"
        fi
        if [ "${#parts[@]}" -ge 2 ] && [ -n "${parts[1]}" ]; then
            start="${parts[1]}"
        fi
        if [ "${#parts[@]}" -ge 3 ] && [ -n "${parts[2]}" ]; then
            end="${parts[2]}"
        fi
    fi
    case "${zmode}" in
        avg|center) ;;
        *) echo "Unknown fp_y z-mode '${zmode}' in token '${token}', expected avg|center" >&2
           return 1 ;;
    esac
    if ! [[ "${start}" =~ ^[0-9]+$ ]]; then
        echo "Invalid start_idx '${start}' in token '${token}', skipping" >&2
        return 1
    fi
    if [ -n "${end}" ] && ! [[ "${end}" =~ ^[0-9]+$ ]]; then
        echo "Invalid end_idx '${end}' in token '${token}', skipping" >&2
        return 1
    fi
    FP_Y_ZMODE+=("${zmode}")
    FP_Y_START_IDX+=("${start}")
    FP_Y_END_IDX+=("${end}")
}

# fp3d_y[:FLAG...][:Z_MODE[:START_IDX[:END_IDX[:Y_IDX]]]] — leading boolean
# FLAG sub-tokens may appear in any order before the positional args:
#   Pi     — top-panel pressure becomes the flow Pi_{rr} = sum_s m_s
#            J_{x,s}^2/(q_s^2 n_s); hoop fold uses -dPi_{rr}/dr +
#            Pi_{phi phi}/r - Pi_{rr}/r (maps to --pi-mode Pi).
#   avgzy  — average over y as well as z instead of slicing at Y_IDX
#            (maps to --avg-zy; Y_IDX is then ignored).
#   pidiag — take Pi_{phi phi} from the momentum_flux_diag_cyl moment's
#            second component instead of the J-flow estimate (--pi).
#   noj    — do not backward-average J / rotM in time (--j).
parse_fp3d_y_token() {
    local token="$1"
    local zmode="avg"
    local start="0"
    local end=""
    local y_idx=""
    local pi_mode="perp"
    local avg_zy="0"
    local pi_diag="0"
    local no_j="0"
    if [ "${token}" != "fp3d_y" ]; then
        local rest="${token#fp3d_y:}"
        IFS=':' read -r -a parts <<< "${rest}"
        local p=0
        # Consume any leading boolean sub-tokens, in any order.
        while [ "${p}" -lt "${#parts[@]}" ]; do
            case "${parts[$p]}" in
                Pi)     pi_mode="Pi" ;;
                avgzy)  avg_zy="1" ;;
                pidiag) pi_diag="1" ;;
                noj)    no_j="1" ;;
                *) break ;;
            esac
            p=$((p + 1))
        done
        if [ "${#parts[@]}" -gt "${p}" ] && [ -n "${parts[$p]}" ]; then
            zmode="${parts[$p]}"
        fi
        if [ "${#parts[@]}" -gt "$((p + 1))" ] \
                && [ -n "${parts[$((p + 1))]}" ]; then
            start="${parts[$((p + 1))]}"
        fi
        if [ "${#parts[@]}" -gt "$((p + 2))" ] \
                && [ -n "${parts[$((p + 2))]}" ]; then
            end="${parts[$((p + 2))]}"
        fi
        if [ "${#parts[@]}" -gt "$((p + 3))" ] \
                && [ -n "${parts[$((p + 3))]}" ]; then
            y_idx="${parts[$((p + 3))]}"
        fi
    fi
    case "${zmode}" in
        avg|center) ;;
        *) echo "Unknown fp3d_y z-mode '${zmode}' in token '${token}', expected avg|center" >&2
           return 1 ;;
    esac
    if ! [[ "${start}" =~ ^[0-9]+$ ]]; then
        echo "Invalid start_idx '${start}' in token '${token}', skipping" >&2
        return 1
    fi
    if [ -n "${end}" ] && ! [[ "${end}" =~ ^[0-9]+$ ]]; then
        echo "Invalid end_idx '${end}' in token '${token}', skipping" >&2
        return 1
    fi
    if [ -n "${y_idx}" ] && ! [[ "${y_idx}" =~ ^[0-9]+$ ]]; then
        echo "Invalid y_idx '${y_idx}' in token '${token}', skipping" >&2
        return 1
    fi
    FP3D_Y_ZMODE+=("${zmode}")
    FP3D_Y_START_IDX+=("${start}")
    FP3D_Y_END_IDX+=("${end}")
    FP3D_Y_Y_IDX+=("${y_idx}")
    FP3D_Y_PI_MODE+=("${pi_mode}")
    FP3D_Y_AVG_ZY+=("${avg_zy}")
    FP3D_Y_PI_DIAG+=("${pi_diag}")
    FP3D_Y_NO_J+=("${no_j}")
}

# fp3d_phi[:Z_MODE[:START_IDX[:END_IDX]]] — phi-averaged, profiles vs r.
parse_fp3d_phi_token() {
    local token="$1"
    local zmode="avg"
    local start="0"
    local end=""
    if [ "${token}" != "fp3d_phi" ]; then
        local rest="${token#fp3d_phi:}"
        IFS=':' read -r -a parts <<< "${rest}"
        if [ "${#parts[@]}" -ge 1 ] && [ -n "${parts[0]}" ]; then
            zmode="${parts[0]}"
        fi
        if [ "${#parts[@]}" -ge 2 ] && [ -n "${parts[1]}" ]; then
            start="${parts[1]}"
        fi
        if [ "${#parts[@]}" -ge 3 ] && [ -n "${parts[2]}" ]; then
            end="${parts[2]}"
        fi
    fi
    case "${zmode}" in
        avg|center) ;;
        *) echo "Unknown fp3d_phi z-mode '${zmode}' in token '${token}', expected avg|center" >&2
           return 1 ;;
    esac
    if ! [[ "${start}" =~ ^[0-9]+$ ]]; then
        echo "Invalid start_idx '${start}' in token '${token}', skipping" >&2
        return 1
    fi
    if [ -n "${end}" ] && ! [[ "${end}" =~ ^[0-9]+$ ]]; then
        echo "Invalid end_idx '${end}' in token '${token}', skipping" >&2
        return 1
    fi
    FP3D_PHI_ZMODE+=("${zmode}")
    FP3D_PHI_START_IDX+=("${start}")
    FP3D_PHI_END_IDX+=("${end}")
}

# fp1d[:START_IDX[:END_IDX]] — z-y-averaged 1D force balance (no z-mode /
# y-idx: always averaged over both z and y).
parse_fp1d_token() {
    local token="$1"
    local start="0"
    local end=""
    if [ "${token}" != "fp1d" ]; then
        local rest="${token#fp1d:}"
        IFS=':' read -r -a parts <<< "${rest}"
        if [ "${#parts[@]}" -ge 1 ] && [ -n "${parts[0]}" ]; then
            start="${parts[0]}"
        fi
        if [ "${#parts[@]}" -ge 2 ] && [ -n "${parts[1]}" ]; then
            end="${parts[1]}"
        fi
    fi
    if ! [[ "${start}" =~ ^[0-9]+$ ]]; then
        echo "Invalid start_idx '${start}' in token '${token}', skipping" >&2
        return 1
    fi
    if [ -n "${end}" ] && ! [[ "${end}" =~ ^[0-9]+$ ]]; then
        echo "Invalid end_idx '${end}' in token '${token}', skipping" >&2
        return 1
    fi
    FP1D_START_IDX+=("${start}")
    FP1D_END_IDX+=("${end}")
}

for d in "${DIAGS[@]}"; do
    case "$d" in
        E|B|J|M|rotM|curlM|dB)
                        FIELDS+=("$d") ;;
        E:*|B:*|J:*|M:*|rotM:*|curlM:*|dB:*)
                        # 3D-slice spec, e.g. "E:z:10" — forwarded as-is to
                        # drift_kinetic_fields.py which parses the axis/idx.
                        FIELDS+=("$d") ;;
        cyl:E|cyl:B|cyl:J|cyl:M|cyl:rotM|cyl:curlM|cyl:dB)
                        FIELDS_CYL+=("${d#cyl:}") ;;
        cyl:E:*|cyl:B:*|cyl:J:*|cyl:M:*|cyl:rotM:*|cyl:curlM:*|cyl:dB:*)
                        FIELDS_CYL+=("${d#cyl:}") ;;
        ions|electrons) PARTICLES+=("$d") ;;
        dBfft)          DBFFT_MODES+=(1) ;;
        dBfft:*)        DBFFT_MODES+=("${d#dBfft:}") ;;
        beta)           RUN_BETA=1 ;;
        energy)         RUN_ENERGY=1 ;;
        energy_compare) ENERGY_COMPARE_TIME_MAX+=("") ;;
        energy_compare:*)
            time_max="${d#energy_compare:}"
            time_re='^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$'
            if ! [[ "${time_max}" =~ ${time_re} ]]; then
                echo "Invalid energy_compare time limit '${time_max}' in '$d', skipping" >&2
                continue
            fi
            ENERGY_COMPARE_TIME_MAX+=("${time_max}")
            ;;
        eq)             EQ_METHODS+=(dec) ;;
        eq:dec)         EQ_METHODS+=(dec) ;;
        eq:pol)         EQ_METHODS+=(pol) ;;
        force|force:*)  parse_force_token "$d" || true ;;
        force_y|force_y:*) parse_force_y_token "$d" || true ;;
        fp_y|fp_y:*)    parse_fp_y_token "$d" || true ;;
        fp3d_y|fp3d_y:*) parse_fp3d_y_token "$d" || true ;;
        fp3d_phi|fp3d_phi:*) parse_fp3d_phi_token "$d" || true ;;
        fp1d|fp1d:*)    parse_fp1d_token "$d" || true ;;
        pcyl|pcyl:*)
            zmode="avg"; idx=""; pmode="full"
            if [ "$d" != "pcyl" ]; then
                rest="${d#pcyl:}"
                IFS=':' read -r -a parts <<< "${rest}"
                # Optional leading "J" selects the J-only mode and shifts
                # the remaining parts down: pcyl:J[:Z_MODE[:IDX]].
                start=0
                if [ "${#parts[@]}" -ge 1 ] && [ "${parts[0]}" = "J" ]; then
                    pmode="J"
                    start=1
                fi
                if [ "${#parts[@]}" -gt "${start}" ] \
                        && [ -n "${parts[$start]}" ]; then
                    zmode="${parts[$start]}"
                fi
                if [ "${#parts[@]}" -gt "$((start + 1))" ] \
                        && [ -n "${parts[$((start + 1))]}" ]; then
                    idx="${parts[$((start + 1))]}"
                fi
            fi
            case "${zmode}" in
                avg|center) ;;
                *) echo "Unknown pcyl z-mode '${zmode}' in '$d', expected avg|center" >&2
                   continue ;;
            esac
            if [ -n "${idx}" ] && ! [[ "${idx}" =~ ^[0-9]+$ ]]; then
                echo "Invalid pcyl idx '${idx}' in '$d', skipping" >&2
                continue
            fi
            PCYL_ZMODE+=("${zmode}")
            PCYL_IDX+=("${idx}")
            PCYL_MODE+=("${pmode}")
            ;;
        profiles)       PROFILES_METHODS+=(dec) ;;
        profiles:dec)   PROFILES_METHODS+=(dec) ;;
        profiles:pol)   PROFILES_METHODS+=(pol) ;;
        density_z)      RUN_DENSITY_Z=1 ;;
        compare_density_z:*) COMPARE_DENSITY_Z+=("${d#compare_density_z:}") ;;
        jspec|jspec:*)
            sub="0"; mode=""
            if [ "$d" != "jspec" ]; then
                rest="${d#jspec:}"
                IFS=':' read -r -a parts <<< "${rest}"
                for p in "${parts[@]}"; do
                    case "${p}" in
                        sub) sub="1" ;;
                        ''|*[!0-9]*)
                            echo "Unknown jspec sub-token '${p}' in '$d', expected 'sub' or an integer mode" >&2
                            continue 2 ;;
                        *) mode="${p}" ;;
                    esac
                done
            fi
            JSPEC_SUB+=("${sub}")
            JSPEC_MODE+=("${mode}")
            ;;
        *) echo "Unknown diagnostic '$d', skipping" >&2 ;;
    esac
done

PY="${PYTHON:-python3}"

if [ "${#FIELDS[@]}" -gt 0 ]; then
    echo "==> fields: ${FIELDS[*]}"
    "${PY}" "${SCRIPT_DIR}/drift_kinetic_fields.py" "${CONFIG}" --fields "${FIELDS[@]}"
fi

if [ "${#FIELDS_CYL[@]}" -gt 0 ]; then
    echo "==> fields (cylindrical): ${FIELDS_CYL[*]}"
    "${PY}" "${SCRIPT_DIR}/drift_kinetic_fields_cyl.py" "${CONFIG}" --fields "${FIELDS_CYL[@]}"
fi

if [ "${#PARTICLES[@]}" -gt 0 ]; then
    echo "==> density: ${PARTICLES[*]}"
    "${PY}" "${SCRIPT_DIR}/drift_kinetic_density.py" "${CONFIG}" --species "${PARTICLES[@]}"
fi

for mode in "${DBFFT_MODES[@]}"; do
    echo "==> dB FFT (mode ${mode}, k_z = 2 pi * ${mode} / L_z)"
    "${PY}" "${SCRIPT_DIR}/drift_kinetic_dB_fft.py" "${CONFIG}" \
        --mode "${mode}" --out-name "dB_fft_mode_${mode}.png"
done

if [ "${RUN_BETA}" -eq 1 ]; then
    echo "==> beta comparison (particle pressure vs displaced field)"
    "${PY}" "${SCRIPT_DIR}/drift_kinetic_beta.py" "${CONFIG}"
fi

if [ "${RUN_ENERGY}" -eq 1 ]; then
    echo "==> energy time series (wE, wB; wK per sort)"
    "${PY}" "${SCRIPT_DIR}/drift_kinetic_energy.py" "${CONFIG}"
fi

for time_max in "${ENERGY_COMPARE_TIME_MAX[@]}"; do
    echo "==> energy comparison across matching ex-runs" \
         "${time_max:+(omega_pe*t <= ${time_max})}"
    if [ -n "${time_max}" ]; then
        "${PY}" "${SCRIPT_DIR}/drift_kinetic_energy_compare.py" "${CONFIG}" \
            --time-max "${time_max}"
    else
        "${PY}" "${SCRIPT_DIR}/drift_kinetic_energy_compare.py" "${CONFIG}"
    fi
done

for method in "${EQ_METHODS[@]}"; do
    echo "==> radial pressure-balance check (method=${method})"
    "${PY}" "${SCRIPT_DIR}/drift_kinetic_equilibrium.py" "${CONFIG}" \
        --method "${method}"
done

for i in "${!FORCE_METHODS[@]}"; do
    method="${FORCE_METHODS[$i]}"
    start="${FORCE_START_IDX[$i]}"
    echo "==> radial momentum-balance check (method=${method}, start_idx=${start})"
    "${PY}" "${SCRIPT_DIR}/drift_kinetic_force.py" "${CONFIG}" \
        --method "${method}" --start-idx "${start}"
done

for i in "${!FORCE_Y_ZMODE[@]}"; do
    zmode="${FORCE_Y_ZMODE[$i]}"
    start="${FORCE_Y_START_IDX[$i]}"
    tavg="${FORCE_Y_TAVG[$i]}"
    tavg_start="${FORCE_Y_TAVG_START[$i]}"
    echo "==> 1D-along-x momentum-balance check on plane-Y" \
         "(z-mode=${zmode}, start_idx=${start}, time-avg=${tavg}," \
         "tavg_start_idx=${tavg_start:-${start}})"
    if [ "${tavg}" = "1" ]; then
        if [ -n "${tavg_start}" ]; then
            "${PY}" "${SCRIPT_DIR}/drift_kinetic_force_y.py" "${CONFIG}" \
                --z-mode "${zmode}" --start-idx "${start}" \
                --time-average --tavg-start-idx "${tavg_start}"
        else
            "${PY}" "${SCRIPT_DIR}/drift_kinetic_force_y.py" "${CONFIG}" \
                --z-mode "${zmode}" --start-idx "${start}" --time-average
        fi
    else
        "${PY}" "${SCRIPT_DIR}/drift_kinetic_force_y.py" "${CONFIG}" \
            --z-mode "${zmode}" --start-idx "${start}"
    fi
done

for method in "${PROFILES_METHODS[@]}"; do
    echo "==> mean radial profiles E_r, J_phi, B_z (method=${method})"
    "${PY}" "${SCRIPT_DIR}/drift_kinetic_profiles.py" "${CONFIG}" \
        --method "${method}"
done

if [ "${RUN_DENSITY_Z}" -eq 1 ]; then
    echo "==> (x, y)-averaged density vs z for every species"
    "${PY}" "${SCRIPT_DIR}/drift_kinetic_density_z.py" "${CONFIG}"
fi

for spec in "${COMPARE_DENSITY_Z[@]}"; do
    # spec is either "<other_name>" or "<other_name>:<t_max_T>", where
    # t_max_T is the upper x-axis limit of the right plot in units of T.
    other_name="${spec%%:*}"
    if [ "${spec}" != "${other_name}" ]; then
        t_max_T="${spec#*:}"
    else
        t_max_T=""
    fi
    other_config="${TEST_DIR}/output/${other_name}/config.json"
    if [ ! -f "${other_config}" ]; then
        echo "compare_density_z: config not found for '${other_name}': ${other_config}" >&2
        continue
    fi
    echo "==> compare ln|delta n_c| vs ${other_name}${t_max_T:+ (t_max = ${t_max_T} T)}"
    out_name="compare_${TEST_NAME}_vs_${other_name}.png"
    if [ -n "${t_max_T}" ]; then
        "${PY}" "${SCRIPT_DIR}/compare_density_z.py" "${CONFIG}" "${other_config}" \
            --filename "${out_name}" --t-max-T "${t_max_T}"
    else
        "${PY}" "${SCRIPT_DIR}/compare_density_z.py" "${CONFIG}" "${other_config}" \
            --filename "${out_name}"
    fi
done

for i in "${!FP_Y_ZMODE[@]}"; do
    zmode="${FP_Y_ZMODE[$i]}"
    start="${FP_Y_START_IDX[$i]}"
    end="${FP_Y_END_IDX[$i]}"
    echo "==> force-balance + pressure on plane-Y" \
         "(z-mode=${zmode}, start_idx=${start}, end_idx=${end:-last})"
    if [ -n "${end}" ]; then
        "${PY}" "${SCRIPT_DIR}/drift_kinetic_force_pressure_y.py" "${CONFIG}" \
            --z-mode "${zmode}" --start-idx "${start}" --end-idx "${end}"
    else
        "${PY}" "${SCRIPT_DIR}/drift_kinetic_force_pressure_y.py" "${CONFIG}" \
            --z-mode "${zmode}" --start-idx "${start}"
    fi
done

for i in "${!FP3D_Y_ZMODE[@]}"; do
    zmode="${FP3D_Y_ZMODE[$i]}"
    start="${FP3D_Y_START_IDX[$i]}"
    end="${FP3D_Y_END_IDX[$i]}"
    y_idx="${FP3D_Y_Y_IDX[$i]}"
    pi_mode="${FP3D_Y_PI_MODE[$i]:-perp}"
    avg_zy="${FP3D_Y_AVG_ZY[$i]:-0}"
    pi_diag="${FP3D_Y_PI_DIAG[$i]:-0}"
    no_j="${FP3D_Y_NO_J[$i]:-0}"
    echo "==> force-balance + pressure from 3D diagnostics on plane-Y" \
         "(pi-mode=${pi_mode}, z-mode=${zmode}, start_idx=${start}," \
         "end_idx=${end:-last}, y_idx=${y_idx:-Ny/2}, avg_zy=${avg_zy}," \
         "pi-diag=${pi_diag}, no-j-shift=${no_j})"
    fp3d_args=(--z-mode "${zmode}" --start-idx "${start}"
               --pi-mode "${pi_mode}")
    if [ -n "${end}" ]; then
        fp3d_args+=(--end-idx "${end}")
    fi
    if [ -n "${y_idx}" ]; then
        fp3d_args+=(--y-idx "${y_idx}")
    fi
    if [ "${avg_zy}" = "1" ]; then
        fp3d_args+=(--avg-zy)
    fi
    if [ "${pi_diag}" = "1" ]; then
        fp3d_args+=(--pi)
    fi
    if [ "${no_j}" = "1" ]; then
        fp3d_args+=(--j)
    fi
    "${PY}" "${SCRIPT_DIR}/drift_kinetic_force_pressure_3D_y.py" "${CONFIG}" \
        "${fp3d_args[@]}"
done

for i in "${!FP3D_PHI_ZMODE[@]}"; do
    zmode="${FP3D_PHI_ZMODE[$i]}"
    start="${FP3D_PHI_START_IDX[$i]}"
    end="${FP3D_PHI_END_IDX[$i]}"
    echo "==> force-balance + pressure from 3D diagnostics, phi-averaged" \
         "(z-mode=${zmode}, start_idx=${start}, end_idx=${end:-last})"
    fp3d_phi_args=(--z-mode "${zmode}" --start-idx "${start}")
    if [ -n "${end}" ]; then
        fp3d_phi_args+=(--end-idx "${end}")
    fi
    "${PY}" "${SCRIPT_DIR}/drift_kinetic_force_pressure_3D_phi.py" "${CONFIG}" \
        "${fp3d_phi_args[@]}"
done

for i in "${!FP1D_START_IDX[@]}"; do
    start="${FP1D_START_IDX[$i]}"
    end="${FP1D_END_IDX[$i]}"
    echo "==> force-balance + pressure, z-y-averaged 1D profile vs x" \
         "(start_idx=${start}, end_idx=${end:-last})"
    fp1d_args=(--start-idx "${start}")
    if [ -n "${end}" ]; then
        fp1d_args+=(--end-idx "${end}")
    fi
    "${PY}" "${SCRIPT_DIR}/drift_kinetic_force_pressure_1D.py" "${CONFIG}" \
        "${fp1d_args[@]}"
done

for i in "${!PCYL_ZMODE[@]}"; do
    zmode="${PCYL_ZMODE[$i]}"
    idx="${PCYL_IDX[$i]}"
    pmode="${PCYL_MODE[$i]}"
    echo "==> cylindrical pressure tensor (Pi_rr, Pi_phiphi, Pi_zz) per species" \
         "(mode=${pmode}, z-mode=${zmode}, idx=${idx:-last})"
    pcyl_args=(--z-mode "${zmode}" --mode "${pmode}")
    if [ -n "${idx}" ]; then
        pcyl_args+=(--idx "${idx}")
    fi
    "${PY}" "${SCRIPT_DIR}/drift_kinetic_pressure_cyl.py" "${CONFIG}" \
        "${pcyl_args[@]}"
done

for i in "${!JSPEC_SUB[@]}"; do
    sub="${JSPEC_SUB[$i]}"
    mode="${JSPEC_MODE[$i]}"
    echo "==> ballistic-spectrum diagnostic for ion sound" \
         "(sub=${sub}, spec_mode=${mode:-1})"
    jspec_args=()
    if [ "${sub}" = "1" ]; then
        jspec_args+=(--subtract-wave)
    fi
    out_name="J_spectrum"
    if [ -n "${mode}" ]; then
        jspec_args+=(--spec-mode "${mode}")
        out_name="${out_name}_m${mode}"
    fi
    if [ "${sub}" = "1" ]; then
        out_name="${out_name}_sub"
    fi
    out_name="${out_name}.png"
    jspec_args+=(--out-name "${out_name}")
    "${PY}" "${SCRIPT_DIR}/drift_kinetic_J_spectrum.py" "${CONFIG}" \
        "${jspec_args[@]}"
done
