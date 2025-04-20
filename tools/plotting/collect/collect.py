import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from lib.common import *

tmin = 0
tmax = int(time / dts) + 1
t_range = mpi_chunks_reduce(np.arange(tmin, tmax, 1))

res_dir = f"{output_path}/collections"
makedirs(res_dir)

def gen_view(path: str, comp: str = None, dof: int = 1) -> FieldView:
    view = FieldView()
    view.path = lambda t: f"{prefix}/{path}/{get_formatted_time(t)}"
    view.region = FieldView.Region(dof, (0, 0, 0), (*data_shape['Z'], dof))
    view.coords = FieldView.Cartesian if not comp in ['r', 'phi'] else FieldView.Cylinder
    view.plane = 'Z'
    view.comp = comp
    return view

type CollectionArrays = dict[str, list[list | FieldView]]

def read(t: int, arrays: CollectionArrays, name: str | list[str]):
    if isinstance(name, str):
        return arrays.get(name)[1].parse(t)
    elif isinstance(name, list) and isinstance(name[0], str):
        data: list[np.ndarray] = []
        for n in name:
            data.append(arrays.get(n)[1].parse(t))
        return data
    return None

def center_avg(d: np.ndarray, w: int = 5):
    c0 = data_shape["Z"][0] // 2
    c1 = data_shape["Z"][1] // 2
    return np.mean(d[c1-w:c1+w, c0-w:c0+w])

def phi_avg(d: np.ndarray):
    return phi_averaged(d, RMAP)

def process_collection(arrays: CollectionArrays, parse: Callable[[int], None], output: Callable[[str], str], shape: tuple = None):
    for t in t_range:
        data = parse(find_correct_timestep(t, t_range))

        for d, arr in zip(data, arrays.values()):
            arr[0].append(d)

    for name, (arr, _) in arrays.items():
        gathered_list = comm.gather(arr, root=0)
        if rank != 0:
          continue

        arr = mpi_chunks_aggregate(gathered_list)
        if shape:
          arr = np.reshape(arr, shape)

        filename = output(name)
        print(f"Saving {'/'.join(filename.split('/')[-2:])}.png")

        np.save(filename, arr)
