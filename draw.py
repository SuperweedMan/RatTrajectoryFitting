#%%
import matplotlib.pyplot as plt
import torch

#%%
states = torch.load('./checkpoint_v12/ckpt_epoch_39')
# states = torch.load('./checkpoint_v9/ckpt_epoch_1998')
#%%
losses = states['losses']
# plt.ion()
line = plt.plot(losses)[0]
# for i in range(2, 25):
# line.set_ydata = losses[:1000]
# plt.pause(10)
plt.show()
