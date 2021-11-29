# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import torch
from pathlib import Path
import math
import torch.nn.functional as F
from torchvision.utils import make_grid


# %%
class Mean:
    def __init__(self):
        self.count = 0
        self.value = 0

    def add(self, newData):
        self.count += 1
        self.value += (newData - self.value) / self.count


class Integration:
    def __init__(self, lower_limit: float, upper_limit: float, resolution: float, cells:int) -> None:
        self.activate_value = None
        self.target_position = None
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.resolution = resolution
        self.measure = int(
            (self.upper_limit - self.lower_limit)/self.resolution)
        self.grid_map = np.zeros((self.measure+1, self.measure+1, cells))
        self.count_map = np.zeros((self.measure+1, self.measure+1) )

    def update_activate_value(self, data: np.ndarray):
        assert len(data.shape) == 2
        self.activate_value = data

    def update_target_position(self, data: np.ndarray):
        assert len(data.shape) == 2
        positions = ((data - self.lower_limit) * (1./self.resolution)).astype(int)
        for idx, position in enumerate(positions):
            position = tuple(position.tolist())
            self.count_map[position] += 1
            self.grid_map[position] += (self.activate_value[idx] -
                                    self.grid_map[position]) / self.count_map[position]


def max_min_scale(data: np.ndarray):
    min = data.reshape(-1,Config.num_of_linear_cell).min(0)
    max = data.reshape(-1,Config.num_of_linear_cell).max(0)
    return (data-min+1e-8) / (max-min)


def GridCell_forward_fn(integrator: Integration):
    def _fn(module, input, output):
        bs, seq, cell_num = output.shape
        integrator.update_activate_value(
            output.reshape(-1, cell_num).detach().clone().cpu().numpy())
    return _fn

def GridCell_pre_forward_fn(integrator: Integration):
    def _fn(module, input):
        input = input[0]
        bs, seq, cell_num = input.shape
        integrator.update_activate_value(
            input.reshape(-1, cell_num).detach().clone().cpu().numpy())
    return _fn

# %%
if __name__ == '__main__':
    from model import RatTrajectoryModel
    from data import Trajectory, PlaceCellEnsemble, HeadCellEnsemble, np_collate_fn
    from config import Config
    from torch.utils.data import DataLoader
    model = RatTrajectoryModel(Config.model_input_size, Config.seq_len)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mode = model.to(device)
    ds = Trajectory(Path(Config.data_root_path))[:10000]
    dl = DataLoader(ds, batch_size=100, shuffle=False, collate_fn=np_collate_fn)
    PCTrans = PlaceCellEnsemble(n_cells=Config.n_place_cells)
    HCTrans = HeadCellEnsemble(n_cells=Config.n_head_cells)
# %%
if __name__ == '__main__':
    import numpy  as np
    from config import Config
    from tqdm import tqdm
    checkpoint = torch.load('./checkpoint_v15/ckpt_epoch_42')
    model.load_state_dict(checkpoint['model'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # fig, ax = plt.subplots(Config.num_of_linear_cell)
    integration = Integration(-1.1, 1.1, 0.2, Config.num_of_linear_cell)
    # integration = Integration(-1.1, 1.1, 0.2, 128)
    handle = model.dropout.register_forward_hook(GridCell_forward_fn(integration))
    # handle = model.linear_layer.register_forward_pre_hook(GridCell_pre_forward_fn(integration))
    # for d in dl:
    for d in tqdm(dl, desc='batch: '):
        # d['target_pos'] = PCTrans(d['target_pos'])
        # d['target_hd'] = HCTrans(d['target_hd'])
        d['init_pos'] = PCTrans(d['init_pos'][:, np.newaxis, :])
        d['init_hd'] = HCTrans(d['init_hd'][:, np.newaxis, :])
        d = {k: torch.from_numpy(v).to(torch.float32) for k, v in d.items()}
        d = {k:v.to(device) for k, v in d.items()}
        output = model(d['ego_vel'], (d['init_pos'], d['init_hd']))

        integration.update_target_position(d['target_pos'].cpu().reshape(-1,2).numpy())

    handle.remove()
#%%    
if __name__ == '__main__':
    # imgs = max_min_scale(integration.grid_map)
    imgs = integration.grid_map
    fig, ax = plt.subplots(16,16, sharex=True, sharey=True)
    ax = ax.flatten()
    # for i in range(128):
    for i in range(Config.num_of_linear_cell):
        ax[i].imshow(imgs[...,i])
    plt.show()
#%%
if __name__ == '__main__':
    fig, ax = plt.subplots(1, sharex=True, sharey=True)
    ax.imshow(integration.grid_map[...,139])
    plt.show()
#%%
# if __name__ == '__main__':
#     import numpy  as np
#     def Guass(x: np.ndarray, sigma: float, mu: np.ndarray):
#         norm = np.linalg.norm((x-mu), axis=-1, keepdims=True)
#         norm2 = np.square(norm)
#         return np.exp(-norm2 / (2*sigma*sigma))

#     fig, ax = plt.subplots(3)
#     x = np.arange(0,2,0.1).repeat((2-0)/0.1).reshape(-1,int((2-0)/0.1))
#     xt = x.T
#     x = np.concatenate((x[...,np.newaxis],xt[...,np.newaxis]), axis=-1)
#     img = Guass(x, sigma=1., mu=1.)
#     ax[0].imshow(img)
    
#     x = np.random.rand(10000,2) * 2
#     integration = Integration(2., 0., 0.1, 2)
#     v = Guass(x, sigma=1., mu=1.)
#     v2 = Guass(x, sigma=1., mu=0.)
#     integration.update_activate_value(np.concatenate((v, v2),axis=-1))
#     integration.update_target_position(x)
#     ax[1].imshow(integration.grid_map[...,0])
#     ax[2].imshow(integration.grid_map[...,1])
#     plt.show()

