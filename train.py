# %%
import enum
import matplotlib.pyplot as plt
import torch
from torch._C import device
from torch.utils.data import DataLoader
from pathlib import Path
from config import Config
from criteria import Criterial, _Criterial, OneCeritial
from data import Trajectory, PlaceCellEnsemble, HeadCellEnsemble, np_collate_fn
from model import RatTrajectoryModel
from utils import LN_forward_fn
import os
import re
import numpy as np
from tqdm import tqdm
from xy_cell import Cells2XYs
import random
# %%
# ds = Trajectory(Path(Config.data_root_path))[:10000]
ds = Trajectory(Path(Config.data_root_path))
dl = DataLoader(ds, batch_size=Config.batch_size,
                shuffle=True, collate_fn=np_collate_fn)
PCTrans = PlaceCellEnsemble(n_cells=Config.n_place_cells)
HCTrans = HeadCellEnsemble(n_cells=Config.n_head_cells)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = RatTrajectoryModel(Config.model_input_size, Config.seq_len)

criterion = Criterial()

# optimizer = torch.optim.RMSprop(model.parameters(), lr=Config.learning_rate)
weight_decay_para_keys = ['linear_layer.weight',
                          'output_layer_pc.weight', 'output_layer_hdc.weight']
weight_decay_list = (param for name, param in model.named_parameters() if name in weight_decay_para_keys)
no_decay_list = (param for name, param in model.named_parameters() if name not in weight_decay_para_keys)

optimizer = torch.optim.Adam(({'params': no_decay_list},
                              {'params': weight_decay_list, 'weight_decay': Config.weight_decay}),
                             lr=Config.learning_rate)

start_epoch = -1
# losses = np.array([]).astype(np.float32)
losses = []
if Config.is_resume:
    checkpoint_dir = Path(Config.check_point_path)
    paths = [path.stem if path.is_file else None for path in checkpoint_dir.iterdir()]
    if len(paths) != 0:  # 存在保存点
        num = [int(re.findall(r'\d+', path)[0]) for path in paths]
        num = max(num)
        checkpoint = torch.load(
            checkpoint_dir / Path('ckpt_epoch_{}'.format(num)))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        losses = checkpoint['losses']

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

model = model.to(device)

# %%
for epoch in tqdm(range(start_epoch+1, Config.epochs), desc='epoch: '):
    model.train()
    for idx, d in tqdm(enumerate(dl), desc='batch: '):
        d['target_pos'] = PCTrans(d['target_pos'])
        d['target_hd'] = HCTrans(d['target_hd'])
        d['init_pos'] = PCTrans(d['init_pos'][:, np.newaxis, :])
        d['init_hd'] = HCTrans(d['init_hd'][:, np.newaxis, :])
        d = {k: torch.from_numpy(v).to(torch.float32) for k, v in d.items()}
        d = {k: v.to(device) for k, v in d.items()}
        if Config.add_noise:
            d['ego_vel'] = d['ego_vel'] + Config.noise_variance * \
                torch.randn(*d['ego_vel'].shape).to(device)
        output = model(d['ego_vel'], (d['init_pos'], d['init_hd']))
        loss = criterion(*output, d['target_pos'], d['target_hd'])
        # loss2 = criterion(d['target_pos'], d['target_hd'], d['target_pos'], d['target_hd'])
        # 更新权重
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), Config.clipping_max_norm, Config.clipping_norm_type)

        optimizer.step()
        # 输出loss
        # tqdm.write("loss: {}  loss2: {}".format(loss, loss2))
        tqdm.write("loss: {}".format(loss))
        # losses = np.concatenate([losses, loss.detach().cpu().numpy().reshape(-1)], axis=0)
        losses.append(loss.detach().cpu().numpy())
    # 收集最好的
    # if losses.size != 0:
    #     if int(losses.argmax()) == losses.size - 1:  # 最新的一次是最好的一次
    #         checkpoint = {
    #             "epoch": epoch,
    #             "optimizer": optimizer.state_dict(),
    #             "model": model.state_dict(),
    #             "losses": losses,
    #         }
    #         train_data_dir = Path(Config.check_point_path)
    #         if not os.path.exists(train_data_dir):
    #             os.mkdir(train_data_dir)
    #         torch.save(checkpoint, train_data_dir /
    #                    Path('best_ckpt_epoch_0'.format(epoch)))
    # 隔批次收集
    if epoch % Config.interval_of_save_weight == 0:
        checkpoint = {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "model": model.state_dict(),
            "losses": losses,
        }
        train_data_dir = Path(Config.check_point_path)
        if not os.path.exists(train_data_dir):
            os.mkdir(train_data_dir)
        torch.save(checkpoint, train_data_dir /
                   Path('ckpt_epoch_{}'.format(epoch)))
        # # 收集激活图片,路径对比图片
        # c2xy = Cells2XYs()
        # # 收集路径对比
        # for x in dl:
        #     x['init_pos'] = PCTrans(x['init_pos'][:, np.newaxis, :])
        #     x['init_hd'] = HCTrans(x['init_hd'][:, np.newaxis, :])
        #     target_pcs = PCTrans(x['target_pos'])
        #     x = {k: torch.from_numpy(v).to(torch.float32) for k, v in x.items()}
        #     x = {k:v.to(device) for k, v in x.items()}
        #     pcs, _ = model(x['ego_vel'], (x['init_pos'], x['init_hd']))
        #     oc = OneCeritial()
        #     _, xys = c2xy.toXYs(pcs[0].detach().clone().cpu().numpy())
        #     _, target_xys = c2xy.toXYs(target_pcs[0])
        #     t_o_loss = oc(pcs.cpu(), torch.from_numpy(target_pcs)).mean()
        #     tqdm.write('targetpc_output_loss: {}'.format(t_o_loss))
        #     xys = np.array(xys)
        #     target_xys = np.array(target_xys)
        #     fig, ax = plt.subplots()
        #     ax.plot(xys[:, 0], xys[:, 1], alpha = 0.5)
        #     ax.plot(target_xys[:,0], target_xys[:,1], alpha = 0.5)
        #     target = x['target_pos'][0].detach().clone().cpu().numpy()
        #     ax.plot(target[:, 0], target[:,1], alpha=0.5, color='red')
        #     plt.savefig(Path(Config.check_point_path) / Path('epoch_{}_paths_{}'.format(epoch, idx)))
        #     break

        # handle = model.linear_layer.register_forward_hook(LN_forward_fn(Path(Config.check_point_path) / Path('HeapMap_{}'.format(epoch))))
        # for x in dl:
        #     # 收集激活
        #     x['target_pos'] = PCTrans(x['target_pos'])
        #     x['target_hd'] = HCTrans(x['target_hd'])
        #     x['init_pos'] = PCTrans(x['init_pos'][:, np.newaxis, :])
        #     x['init_hd'] = HCTrans(x['init_hd'][:, np.newaxis, :])
        #     x = {k: torch.from_numpy(v).to(torch.float32) for k, v in x.items()}
        #     x = {k:v.to(device) for k, v in x.items()}
        #     model(x['ego_vel'], (x['init_pos'], x['init_hd']))
        #     break
        # handle.remove()
