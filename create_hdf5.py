from tqdm import tqdm
from itertools import islice
import h5py
from nsm.utils import infinite_graphs

gen = infinite_graphs()

exponent = 4
n_graphs = 10 ** exponent
with h5py.File("data/deleteme.h5", "w") as f:
    for i, graph in tqdm(enumerate(islice(gen, n_graphs)), total=n_graphs):
        # breakpoint()
        name = f"graph{i:0{exponent}}"
        grp = f.create_group(name)
        for key, value in graph._asdict().items():
            grp[key] = value
