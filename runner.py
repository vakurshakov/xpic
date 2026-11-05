#!/usr/bin/env python3
import os, json

pool = [
    ("eccapfim", 1000, 1.0),
    ("ecsimcorr", 1000, 1.0),

    ("eccapfim", 200, 0.5),
    ("ecsimcorr", 200, 0.5),
]

tol = 1e-7

dt = 0.5  # 1 / w_pe
dts = 50 * dt    # 1 / w_pe
tau = 1000       # 1 / w_pe 
dtb = tau / 2    # 1 / w_pe
time = 6 * tau   # 1 / w_pe

B0 = 0.1978

mi_me = 16

Te = 2   # KeV
Ti = 10  # KeV

out_result = None
out_config = None
config = None

os.makedirs("results", exist_ok=True)
os.makedirs("configs", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def gen_out_paths():
    global out_result, out_config

    name = f"{sim}_dx{dx:.1f}_dt{dt:.1f}_Np{Np:d}_mi{mi_me:d}_tol{tol:.0e}"

    out_result = f"results/{name}"
    out_config = f"configs/{name}.json"
    # i = 1
    # 
    # while os.path.exists(out_result):
    #     out_result = f"results/{name}_new{i}"
    #     out_config = f"configs/{name}_new{i}.json"
    #     i += 1

def gen_out_dirs():
    try:
        os.makedirs(out_result) # , exist_ok=True)
        os.makedirs(os.path.dirname(out_config), exist_ok=True)
    except Exception as e:
        print(f"An error occurred: {e}\n") 

def gen_config():
    global config

    def box(r):
        return {
            "min": [ f"{Nx // 2 - r} [dx]", 0, 0 ],
            "max": [ f"{Nx // 2 + r} [dx]", f"{Ny} [dy]", f"{Nz} [dz]" ],
        }
    
    g_damp = { "geometry": { "name": "BoxGeometry", **box(r_damp) } }

    def distribution_moment(sort): 
        return {
            "diagnostic": "DistributionMoment",
            "particles": sort,
            "moment": "momentum_flux",
            "region": {
                "start": [ f"0 [dx]", f"0 [dy]", f"{Nz // 2} [dz]" ],
                "size":  [ f"{Nx} [dx]", f"{Ny} [dy]", f"1 [dz]" ]
            },
            "out_dir": f"{sort}/momentum_Z{Nz // 2}",
        }

    config = {
        "Simulation": sim,
        "OutputDirectory": out_result,

        "Geometry": {
            "x": f"{Nx:d} [dx]",
            "y": f"{Ny:d} [dy]",
            "z": f"{Nz:d} [dz]",
            "t": f"{round(time / dt):d} [dt]",
            "dx": round(dx, 2),
            "dy": round(dy, 2),
            "dz": round(dz, 2),
            "dt": round(dt, 2),
            "diagnose_period": f"{round(dts / dt)} [dt]",
            "da_boundary_x": "DM_BOUNDARY_GHOSTED",
            "da_boundary_y": "DM_BOUNDARY_PERIODIC",
            "da_boundary_z": "DM_BOUNDARY_PERIODIC",
        },

        "Particles": [
            { "sort_name": "ions", "Np": Np, "n": 1, "q": +1, "m": mi_me, "T": Ti },
            { "sort_name": "electrons", "Np": Np, "n": 1, "q": -1, "m": 1, "T": Te },
        ],

        "Presets": [
            { 
                "command": "SetMagneticField",
                "setter": { "name": "SetUniformField", "value": [0, 0, B0] },
            }
        ],

        "StepPresets": [
            {
                "command": "FieldsDamping",
                "damping_coefficient": 0.8,
                **g_damp,
            },
            {   
                "command": "RemoveParticles",
                "particles": "ions",
                **g_damp,
            },
            {   
                "command": "RemoveParticles",
                "particles": "electrons",
                **g_damp,
            },
            {
                "command": "InjectParticles",
                "ionized": "ions",
                "ejected": "electrons",
                "injection_start": "0 [dt]",
                "injection_end":  f"{round(time / dt)} [dt]",
                "tau":            f"{round(tau  / dt)} [dt]",
                "coordinate": { "name": "CoordinateInBox", **box(r_inj) },
                "momentum_i": { "name": "MaxwellianMomentum", "tov": True },
                "momentum_e": { "name": "MaxwellianMomentum", "tov": True },
            }
        ],
        
        "Diagnostics": [
            { "diagnostic": "LogView", "level": "EachTimestep" },
            { "diagnostic": "LogView", "level": "DiagnosePeriodAvg" },
            { "diagnostic": "LogView", "level": "AllTimestepsSummary" },

            { "diagnostic": "FieldViewZAvg", "field": "E" },
            { "diagnostic": "FieldViewZAvg", "field": "B" },
            
            { "diagnostic": "FieldViewZAvg", "field": "ions/J" },
            { "diagnostic": "FieldViewZAvg", "field": "ions/rho" },
            { "diagnostic": "FieldViewZAvg", "field": "electrons/J" },
            { "diagnostic": "FieldViewZAvg", "field": "electrons/rho" },

            distribution_moment("ions"),
            distribution_moment("electrons")
        ],

        "SimulationBackup": {
            "diagnose_period": f"{round(dtb / dt)} [dt]",
        }
    }

def submit():
    with open(out_config, "w") as f:
        json.dump(config, f, indent=4)

    if sim == "eccapfim":
        options = f"-snes_type anderson -snes_atol {tol:.0e} -snes_rtol {tol:.0e} -snes_stol {tol:.0e}"
    elif sim == "ecsimcorr":
        options = f"-ksp_atol {tol:.0e} -ksp_rtol {tol:.0e}"

    os.system(f"qsub ./run.sh {out_config} {out_result}/output {options}")
    
if __name__ == "__main__":
    for sim, Np, d in pool:
        dx = d  # c / w_pe
        dy = d  # c / w_pe
        dz = d  # c / w_pe

        Lx = 200     # c / w_pe
        Ly = 40      # c / w_pe
        Lz = 5 * dz  # c / w_pe

        Nx = round(Lx / dx)  # cells
        Ny = round(Ly / dy)  # cells
        Nz = round(Lz / dz)  # cells
        buff = 8             # cells

        r_inj = round(12.5 / dx)  # cells
        r_damp = Nx // 2 - buff   # cells

        gen_out_paths()
        gen_out_dirs()
        gen_config()
        submit()
