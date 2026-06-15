import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from lib.common import *

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
