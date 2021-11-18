#%%
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
import math
import torch.nn.functional as F
from torchvision.utils import make_grid

#%%
if __name__ == '__main__':
    from model import RatTrajectoryModel
    from data import Trajectory, PlaceCellEnsemble, HeadCellEnsemble, np_collate_fn
    from config import Config
    from torch.utils.data import DataLoader
    from xy_cell import Cells2XYs
    from criteria import OneCeritial
    from tqdm import tqdm

    
    model = RatTrajectoryModel(Config.model_input_size, Config.seq_len)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mode = model.to(device)
    ds = Trajectory(Path(Config.data_root_path))[:100]
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=np_collate_fn)
    PCTrans = PlaceCellEnsemble(n_cells=Config.n_place_cells)
    HCTrans = HeadCellEnsemble(n_cells=Config.n_head_cells)

    ckp_version = 12
    ckp_epochs= 39
    ckp_path = './checkpoint_v12/ckpt_epoch_39'
    ckp = torch.load(Path(ckp_path))
    model.load_state_dict(ckp['model'])

    c2xy = Cells2XYs()
    # 收集路径对比
    for idx, x in enumerate(dl):
        x['init_pos'] = PCTrans(x['init_pos'][:, np.newaxis, :])
        x['init_hd'] = HCTrans(x['init_hd'][:, np.newaxis, :])
        target_pcs = PCTrans(x['target_pos'])
        x = {k: torch.from_numpy(v).to(torch.float32) for k, v in x.items()}
        x = {k:v.to(device) for k, v in x.items()}
        if Config.add_noise:
            x['ego_vel'] = x['ego_vel'] + Config.noise_variance*torch.randn(*x['ego_vel'].shape).to(device)
        pcs, _ = model(x['ego_vel'], (x['init_pos'], x['init_hd']))
        oc = OneCeritial()
        _, xys = c2xy.toXYs(pcs[0].detach().clone().cpu().numpy())
        # _, target_xys = c2xy.toXYs(target_pcs[0])
        # t_o_loss = oc(pcs.cpu(), torch.from_numpy(target_pcs)).mean()
        # tqdm.write('targetpc_output_loss: {}'.format(t_o_loss))
        xys = np.array(xys)
        # target_xys = np.array(target_xys)
        fig, ax = plt.subplots()
        ax.plot(xys[:, 0], xys[:, 1], alpha = 0.5)
        # ax.plot(target_xys[:,0], target_xys[:,1], alpha = 0.5)
        target = x['target_pos'][0].detach().clone().cpu().numpy()
        ax.plot(target[:, 0], target[:,1], alpha=0.5, color='red')
        plt.savefig(Path('./fitting_path')/ Path('v{}_epoch{}_tra{}'.format(ckp_version, ckp_epochs, idx))) 