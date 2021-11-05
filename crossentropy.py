# %%
import torch
from data import Trajectory, PlaceCellEnsemble, HeadCellEnsemble, np_collate_fn
from config import Config
from torch.utils.data import DataLoader
from pathlib import Path
from torch.nn import CrossEntropyLoss
import numpy as np
# %%
ds = Trajectory(Path('../GridCellDataset/npy'))
dl = torch.utils.data.DataLoader(
    ds[:2], batch_size=3, collate_fn=np_collate_fn)
PCTrans = PlaceCellEnsemble(n_cells=Config.n_place_cells)
HCTrans = HeadCellEnsemble(n_cells=Config.n_head_cells)

l= CrossEntropyLoss()
#%%
for i in dl:
    a = torch.from_numpy(HCTrans(i['target_hd'])).to(torch.double)
    a = torch.from_numpy(HCTrans(i['target_hd'])).to(torch.float32)
    a_label = torch.argmax(a, -1)
    a = a.permute(0, 2, 1)
    print(l(a, a_label))
    
    
    