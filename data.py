# %%
import pathlib
from matplotlib.pyplot import axis
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Iterable
from config import Config

# %%

def np_collate_fn(x):
    values = list(zip(*[i.values() for i in x]))
    values = [np.array(v) for v in values]
    keys = list(x[0].keys())
    return dict(zip(keys, values))



class Trajectory:
    def __init__(self, root_path: pathlib.Path) -> None:
        data_paths = list(root_path.rglob('*.npy'))
        self.data = [np.load(path, allow_pickle=True) for path in data_paths]
        self.data = np.concatenate(self.data, axis=0)
        self.data.dtype

    def __getitem__(self, index) -> np.ndarray:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class MyTransform:
    def __init__(self, ops: Iterable) -> None:
        self.ops = ops

    def __call__(self, *args):
        for op in self.ops:
            args = op(*args)
        return args


class Add:
    def __init__(self) -> None:
        pass

    def __call__(self, *args):
        return [i + 1 for i in args]


class PlaceCellEnsemble:
    def __init__(self, n_cells, seed: int = 1, pos_min: int = -5, pos_max: int = 5, stdev: float = 0.35) -> None:
        rs = np.random.RandomState(seed)
        self.n_cells = n_cells
        self.means = rs.uniform(pos_min, pos_max, size=(self.n_cells, 2))
        self.variances = np.ones_like(self.means) * stdev**2

    def __call__(self, PlaceCellValues: np.ndarray):
        v = PlaceCellValues
        # [bs, Tra, 2], [ncell, 2] -> [bs, Tra, ncell, 2]
        diff = v[:, :, np.newaxis, :] - self.means[np.newaxis, np.newaxis, ...]
        # [bs, tra, ncell, 2] -> [bs, tra, ncell]
        unnor_logp = -0.5 * \
            np.sum((diff**2)/self.variances, axis=-1, keepdims=False)  #
        log_posteriors = unnor_logp - \
            np.log(np.sum(np.exp(unnor_logp), 2, keepdims=True))
        # [bs, tra, ncell]
        log_posteriors = F.softmax(torch.from_numpy(log_posteriors), dim=-1).numpy()
        return log_posteriors


class HeadCellEnsemble:
    def __init__(self, n_cells, seed: int = 1, concentration=20) -> None:
        rs = np.random.RandomState(seed)
        self.means = rs.uniform(-np.pi, np.pi, (n_cells))
        self.kappa = np.ones_like(self.means) * concentration

    def __call__(self, HeadCellValues: np.ndarray):
        # [bs, tra, 1]
        unnor_logpdf = self.kappa * \
            np.cos(HeadCellValues - self.means[np.newaxis, np.newaxis, :])
        log_posteriors = unnor_logpdf - \
            np.log(np.sum(np.exp(unnor_logpdf), 2, keepdims=True))
        log_posteriors = F.softmax(torch.from_numpy(log_posteriors), dim=-1).numpy()
        return log_posteriors

# %%
if __name__ == '__main__':
    from model import RatTrajectoryModel
    model = RatTrajectoryModel(3, Config.seq_len)
    ds = Trajectory(Path('../GridCellDataset/npy'))
    dl = torch.utils.data.DataLoader(ds, batch_size=1000, collate_fn=np_collate_fn)
    PCTrans = PlaceCellEnsemble(n_cells=Config.n_place_cells)
    HCTrans = HeadCellEnsemble(n_cells=Config.n_head_cells)
#%%
if __name__ == '__main__':
    for d in dl:
        break
        d['target_pos'] = PCTrans(d['target_pos'])
        d['target_hd'] = HCTrans(d['target_hd'])
        d['init_pos'] = PCTrans(d['init_pos'][:, np.newaxis, :])
        d['init_hd'] = HCTrans(d['init_hd'][:, np.newaxis, :])
        d = {k: torch.from_numpy(v).to(torch.float32) for k, v in d.items()}
        # output = model(d['ego_vel'], (d['init_pos'], d['init_hd']))
    # input = np.random.randn(3, 100, 1)
    # print(transformer(input))
#%%
if __name__ == '__main__':
    import numpy as np
    from sincos2radian import sincos2radians
    dl = torch.utils.data.DataLoader(ds, batch_size=1000, collate_fn=np_collate_fn)
    for d in dl:
        break
    position  = d['target_pos'][0][:10]
    velocity = d['ego_vel'][0][:10][...,0]
    position_error = np.concatenate([d['init_pos'][0][None], position], axis=0) - np.concatenate([position, np.array([[0, 0]])], axis=0)
    value = np.linalg.norm(position_error,axis=-1)

    angle_v = d['target_hd'][0][:10]
    
    sincos_angle_v = sincos2radians(d['ego_vel'][0][:10][...,1:])
# %%
    angle_v_error = np.concatenate([np.array([[0]]), angle_v], axis=0) - np.concatenate([angle_v, np.array([[0]])], axis=0)
#%%
import math
def angle2worldangle(a):
    return -(a-(math.pi/2.))/math.pi *180.