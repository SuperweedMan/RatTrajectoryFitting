# %%
import torch
import torch.nn as nn
# %%

class OneCeritial:
    def __init__(self) -> None:
        # self.crossEngropyL = nn.CrossEntropyLoss()
        # self.log_func = nn.LogSoftmax(dim=-1)
        pass

    def __call__(self, pre_pc: torch.tensor, target_pc: torch.tensor):
        # target_hd = torch.argmax(target_hd, dim=-1).reshape(-1)
        # target_pc = torch.argmax(target_pc, dim=-1).reshape(-1)
        # target_pc = target_pc.reshape(-1, target_pc.shape[-1])

        # pre_pc = pre_pc.reshape(-1, pre_pc.shape[-1])

        pc_loss = torch.multiply(
            target_pc, -torch.log(pre_pc+1e-8)).sum(-1)
        return pc_loss


class Criterial:
    def __init__(self) -> None:
        # self.crossEngropyL = nn.CrossEntropyLoss()
        # self.logsoftmax_func = nn.LogSoftmax(dim=-1)
        pass

    def __call__(self, pre_pc: torch.tensor, pre_hd: torch.tensor, target_pc: torch.tensor, target_hd: torch.tensor):
        # target_hd = torch.argmax(target_hd, dim=-1).reshape(-1)
        # target_pc = torch.argmax(target_pc, dim=-1).reshape(-1)
        num = 1
        for i in list(target_hd.shape[:-1]):
            num *= i
        target_hd = target_hd.reshape(-1, target_hd.shape[-1])
        target_pc = target_pc.reshape(-1, target_pc.shape[-1])

        pre_pc = pre_pc.reshape(-1, pre_pc.shape[-1])
        pre_hd = pre_hd.reshape(-1, pre_hd.shape[-1])

        # pc_loss = self.crossEngropyL(pre_pc, target_pc)
        # hd_loss = self.crossEngropyL(pre_hd, target_hd)
        pc_loss = torch.multiply(
            target_pc, -torch.log(pre_pc+1e-8)).sum() / num
        hd_loss = torch.multiply(
            target_hd, -torch.log(pre_hd+1e-8)).sum() / num
        total_loss = pc_loss+hd_loss
        return total_loss


class _Criterial:
    def __init__(self) -> None:
        self.crossEngropyL = nn.CrossEntropyLoss()
        # self.logsoftmax_func=nn.LogSoftmax(dim=-1)

    def __call__(self, pre_pc: torch.tensor, pre_hd: torch.tensor, target_pc: torch.tensor, target_hd: torch.tensor):
        target_hd = torch.argmax(target_hd, dim=-1)
        target_pc = torch.argmax(target_pc, dim=-1)
        # target_hd = target_hd.reshape(-1, target_hd.shape[-1])
        # target_pc = target_pc.reshape(-1, target_pc.shape[-1])

        pre_pc = pre_pc.permute(0, 2, 1)
        pre_hd = pre_hd.permute(0, 2, 1)

        pc_loss = self.crossEngropyL(pre_pc, target_pc)
        hd_loss = self.crossEngropyL(pre_hd, target_hd)
        # pc_loss = torch.multiply(target_pc ,-self.logsoftmax_func(pre_pc)).sum()
        # hd_loss = torch.multiply(target_hd, -self.logsoftmax_func(pre_hd)).sum()
        total_loss = pc_loss+hd_loss
        return total_loss
    


# %%
if __name__ == '__main__':
    from data import Trajectory, np_collate_fn, PlaceCellEnsemble, HeadCellEnsemble
    from config import Config
    from data import Trajectory
#%%
if __name__ == '__main__':
    from torch.nn.functional import softmax

    target = torch.randn((2, 10, 25))
    target = softmax(target, dim=-1)

    input = torch.randn((2, 10, 25))
    input = softmax(input, dim=-1)
    c1 = _Criterial()
    c2 = Criterial()
    loss1 = c1(input, input, target, target)
    loss2 = c2(input, input, target, target)
    
    loss3 = c1(target, target, target, target)
    loss4 = c2(target, target, target, target)
    input = target
    input[0][1][2] += 0.7
    input = softmax(input, dim=-1)
    loss5 = c1(input, input, target, target)
    loss6 = c2(input, input, target, target)
    print(loss1, loss2, loss3, loss4, loss5, loss6)
    # %%
if __name__ == '__main__':
    from model import RatTrajectoryModel
    from data import Trajectory, np_collate_fn, PlaceCellEnsemble, HeadCellEnsemble
    from torch.utils.data import DataLoader
    import numpy as np
    from pathlib import Path
    from config import Config
    model = RatTrajectoryModel(Config.model_input_size, Config.seq_len)
    ds = Trajectory(Path(Config.data_root_path))
    dl = DataLoader(ds[:3], batch_size=3, collate_fn=np_collate_fn)
# %%
if __name__ == '__main__':
    criterion = Criterial()
    PCTrans = PlaceCellEnsemble(n_cells=Config.n_place_cells)
    HCTrans = HeadCellEnsemble(n_cells=Config.n_head_cells)
    for d in dl:
        d['target_pos'] = PCTrans(d['target_pos'])
        d['target_hd'] = HCTrans(d['target_hd'])
        d['init_pos'] = PCTrans(d['init_pos'][:, np.newaxis, :])
        d['init_hd'] = HCTrans(d['init_hd'][:, np.newaxis, :])
        d = {k: torch.from_numpy(v).to(torch.float32) for k, v in d.items()}
        output = model(d['ego_vel'], (d['init_pos'], d['init_hd']))
        loss = criterion(*output, d['target_pos'], d['target_hd'])
        print(loss)
