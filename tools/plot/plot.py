import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from lib.common import *
from lib.constants import *

def process_plots(fig, out_subdir, time, plots, callback):
    out_dir = f"{const.out_dir}/{out_subdir}"; makedirs(out_dir)

    for t in mpi_consecutive_t_range(const.pts, const.pte, const.pto):
        figname = f"{out_dir}/{str(t // const.pto).zfill(4)}.png"
        figstring = f"{figname} {t} [dts] {t * const.dt / const.tau:.2f} [tau]"

        if os.path.exists(figname):
            print(f"File exists: {figstring}")
            continue

        try:
            callback(t)
            print(f"Processing: {figstring}")

            fig.suptitle(time(t), x=0.50, y=0.99, bbox=bbox, fontsize=labelsize)
            fig.tight_layout(rect=(0, 0, 1, 0.99))
            fig.savefig(figname)

            for plot in plots:
                plot.clear()

        except FileNotFoundError:
            print(f"File not found: {figstring}")
    
    comm.barrier()

    if rank == 0:
        vid_dir = f"{const.out_dir}/video"; makedirs(vid_dir)
        os.system(f"ffmpeg -y -i {out_dir}/%04d.png -r 15 {vid_dir}/{out_subdir}.mp4")

    comm.barrier()
    

def time_wpe(t):
    return f"$\\omega_{{ pe }}\\,t = {t * const.dt:.3f}$"

def time_wce(t):
    return f"$\\Omega_e\\,t = {t * const.dt * const.B0:.3f}$"

def time_wci(t):
    return f"$\\Omega_i\\,t = {t * const.dt * (const.B0 / const.mi_me):.3f}$"

def time_tau(t):
    return f"$t/\\tau = {(t * const.dt / const.tau):.2f}$"