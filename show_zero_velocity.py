#%%
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import torch
from pathlib import Path
from model import RatTrajectoryModel
from data import Trajectory, PlaceCellEnsemble, HeadCellEnsemble, np_collate_fn
from config import Config
from torch.utils.data import DataLoader
from tqdm import tqdm
#%%
if __name__ == '__main__':
    ds = Trajectory(Path(Config.data_root_path))
#%%
if __name__ == '__main__':
    dl = DataLoader(ds, batch_size=1000, shuffle=False, collate_fn=np_collate_fn)
    for d in tqdm(dl, desc='batch: '):
        v = d['ego_vel']
        positions = d['target_pos']
        xys = positions[v[...,0]<0.01]
        break
    fig, ax = plt.subplots(1)
    ax.scatter(xys[:,0], xys[:,1])
    plt.show()