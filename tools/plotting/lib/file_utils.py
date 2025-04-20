import os
from mpi4py import MPI

# Files manipulation utilities

def makedirs(dirname):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if not os.path.exists(dirname) and rank == 0:
        os.makedirs(dirname, exist_ok=True)
    comm.barrier()

def dump(prefix, name, units, t, data):
    makedirs(prefix)
    with open(f"{prefix}/{name}_{t}.txt", "w") as f:
        f.write(f"Grid {name} [{units}]\n")
        for i, d in enumerate(data):
            f.write(f"{i} {d}\n")
