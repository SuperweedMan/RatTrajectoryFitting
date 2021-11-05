#%%
import matplotlib.pyplot as plt
import torch

#%%
states = torch.load('./checkpoint/ckpt_epoch_36')
#%%
losses = states['losses']
plt.plot(losses)
plt.show()