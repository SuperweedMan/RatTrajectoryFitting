# %%
'''
将激活序列转为xy坐标值
'''
import torch
from torch.utils.data import DataLoader
import numpy as np
from criteria import OneCeritial
import matplotlib.pyplot as plt
from data import Trajectory, PlaceCellEnsemble, HeadCellEnsemble, np_collate_fn
from pathlib import Path
from config import Config
# %%


class Cells2XYs:
    def __init__(self, up_limit: float = 2., down_limit: float = -2, interval: float = 0.01) -> None:
        self.up_limit = up_limit
        self.down_limit = down_limit
        self.interval = interval
        self.PCtrans = PlaceCellEnsemble(n_cells=Config.n_place_cells)
        self.criterial = OneCeritial()

    def toXYs(self, Cells: np.ndarray):
        '''
        将输入的np数组[seqs, n_places],转为xy坐标系[seqs, 2]
        '''
        x = np.arange(self.down_limit, self.up_limit, self.interval).repeat(int(
            (self.up_limit-self.down_limit)/self.interval)).reshape(-1, int((self.up_limit-self.down_limit)/self.interval))
        y = x.T
        # xyaxis [h, w, 2]
        xyaxis = np.concatenate([x[:, :, np.newaxis],
                                y[:, :, np.newaxis]], axis=-1)
        h, w, _ = xyaxis.shape
        xycells = self.PCtrans(xyaxis).reshape(-1,Config.n_place_cells)
        l = []
        d = []
        xycells = torch.from_numpy(xycells)
        Cells = torch.from_numpy(Cells)
        for cell in Cells:
            xyv = self.criterial(xycells, cell)
            idx = np.argmin(xyv.numpy())
            xyv = xyv.reshape(h,w, -1)
            h_idx = int(idx / w) *0.01 + self.down_limit
            w_idx = int(idx % w)*0.01 + self.down_limit
            l.append(xyv)
            d.append((h_idx, w_idx))
        return l, d
#%%
if __name__ == '__main__':
    PCTrans = PlaceCellEnsemble(n_cells=Config.n_place_cells)
    # HCTrans = HeadCellEnsemble(n_cells=Config.n_head_cells)
    ds = Trajectory(Path(Config.data_root_path))[:100]
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=np_collate_fn)
# %%
if __name__ == '__main__':
    for d in dl:
        xys = d['target_pos'][0][:3]
        cells = PCTrans(d['target_pos'])[0][:3]
        c2x = Cells2XYs(up_limit=2, down_limit=-2)
        V, D = c2x.toXYs(cells)
        fig, ax = plt.subplots()
        ax.plot(xys[:, 0], xys[:, 1])
        plt.show()
        for v in V:
            fig, ax = plt.subplots()
            ax.imshow(v, cmap=plt.get_cmap('hot'))
            plt.xlim(0, 400)
            plt.ylim(0,400)
            plt.show()
        for d in D:
            print(d)
        break
