import numpy as np

from mpi4py import MPI

# MPI utilities

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def mpi_chunks_reduce(a):
    length = len(a)
    chunk = length // size
    remain = length % size
    prev = 0

    if rank < remain:
        prev = (chunk + 1) * rank
        chunk += 1
    else:
        prev = chunk * rank + remain
    return a[prev: prev + chunk]

def mpi_chunks_aggregate(a):
    result = np.array([])
    for s in a:
        result = np.append(result, s)
    return result

def mpi_consecutive_t_range(tmin: int, tmax: int, offset: int):
    return np.arange(tmin + rank * offset, tmax + 1, size * offset)

def mpi_consecutive_reduce(a):
    return a[rank::size]

def mpi_consecutive_aggregate(a):
    result = np.array([])
    maxl = max(map(lambda i: len(i), a))
    for i in range(0, maxl):
        for s in a:
            if i < len(s):
                result = np.append(result, s[i])
    return result
