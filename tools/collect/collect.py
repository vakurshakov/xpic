import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from lib.common import *
from lib.constants import *

def read(file_t, c, cd):
    data_shape = [const.Ny, const.Nx, cd]
    with open(file_t, "rb") as file:
        raw = np.fromfile(file, dtype=np.float32, count=np.prod(data_shape), offset=0)
        raw = np.reshape(raw, data_shape)
    return raw[:,:,c]

def read_E(c): return lambda t: read(f"{const.in_dir}/E_ZAvg/{str(t // const.Ndts).zfill(4)}", c, 3)
def read_B(c): return lambda t: read(f"{const.in_dir}/B_ZAvg/{str(t // const.Ndts).zfill(4)}", c, 3)
def read_J(s, c): return lambda t: read(f"{const.in_dir}/{s}/J_ZAvg/{str(t // const.Ndts).zfill(4)}", c, 3)
def read_P(s, c): return lambda t: read(f"{const.in_dir}/{s}/momentum_Z2/{str(t // const.Ndts).zfill(4)}", c, 6)
def read_n(s):    return lambda t: read(f"{const.in_dir}/{s}/rho_ZAvg/{str(t // const.Ndts).zfill(4)}", 0, 1) * (1 if s == "ions" else -1)

def center_avg(d: np.ndarray, w: int = 5):
    c0 = d.shape[0] // 2
    c1 = d.shape[1] // 2
    return np.mean(d[c0-w:c0+w, c1-w:c1+w])

def phi_avg(d: np.ndarray):
    return phi_averaged(d, const.rmap)

def process_collection(arrays, parse, output, shape=None):
    for t in mpi_chunks_reduce(np.arange(const.pts, const.pte, const.pto)):
        tstring = f"{t} [dts] {t * const.dt / const.tau:.2f} [tau]"

        try:
            data = parse(t)
            print(f"Processing {tstring}")

            for d, arr in zip(data, arrays.values()):
                arr.append(d)

        except FileNotFoundError:
            print(f"File not found: {tstring}")

    for i, (name, arr) in enumerate(arrays.items()):
        gathered_list = comm.allgather(arr)
        if rank != i % size:
          continue

        arr = mpi_chunks_aggregate(gathered_list)
        if shape:
          arr = np.reshape(arr, shape)

        filename = output(name)
        print(f"Saving {'/'.join(filename.split('/')[-4:])}.png")

        makedirs(os.path.dirname(filename))
        np.save(filename, arr)
