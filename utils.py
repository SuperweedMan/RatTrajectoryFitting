# %%
from types import CellType
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from pathlib import Path 
from torchvision.utils import make_grid


def show_np_matrix(data: np.ndarray):
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='plasma_r')
    # fig.colorbar(im, pad=0.03)
    fig.tight_layout()
    plt.show()


def save_np_matrix(data: np.ndarray, path:Path, dpi=None):
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='plasma_r')
    # fig.colorbar(im, pad=0.03)
    fig.tight_layout()
    if dpi is None:
        plt.savefig(path, dpi=900)
    else:
        plt.savefig(path, dpi=dpi)

def LN_forward_fn(save_path:Path):
    def _fn(module, input, output):
        bs, seq, cell_num = output.shape
        w = h = int(math.sqrt(cell_num))
        data = output.detach().clone().cpu().reshape(-1, cell_num)
        np.save(save_path, data.numpy())
        data = make_grid(data.reshape(-1, 1, h, w), nrow=50).numpy()
        data = np.transpose(data, (1,2,0))
        save_np_matrix(data, save_path)
    return _fn

# %%
if __name__ == '__main__':
    data = torch.from_numpy(np.random.rand(2, 2, 30 * 30))
    # show_np_matrix(data)
    # save_np_matrix(data, Path('./'))
    fn = LN_forward_fn(Path('./'))
    fn(None, None, data)