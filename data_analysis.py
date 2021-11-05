#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
from data import Trajectory, np_collate_fn, PlaceCellEnsemble, HeadCellEnsemble
from config import Config
from torch.utils.data import DataLoader
from pathlib import Path
#%%
ds = Trajectory(Path(Config.data_root_path))[:5000]
dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=np_collate_fn)
#%%
PCTrans = PlaceCellEnsemble(n_cells=Config.n_place_cells)
HCTrans = HeadCellEnsemble(n_cells=Config.n_head_cells)

for d in dl:
    d['target_pos'] = PCTrans(d['target_pos'])
    d['target_hd'] = HCTrans(d['target_hd'])
    d['init_pos'] = PCTrans(d['init_pos'][:, np.newaxis, :])
    d['init_hd'] = HCTrans(d['init_hd'][:, np.newaxis, :])
    # print(d['target_pos'])
    test = np.zeros((1,100,256))
    test[:,:,0]=1
    var_test = np.var(test, axis=-1)
    var = np.var(d['target_pos'], axis=-1)
    break