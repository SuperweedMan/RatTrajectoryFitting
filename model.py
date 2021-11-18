# %%
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from config import Config


# %%
LSTM_HIDDEN_UNITS = 128
NUM_OF_LSTM_HIDDEN_LAYERS = 1
# NUM_OF_LINEAR_CELLS = 256
# NUM_OF_LINEAR_CELLS = 512
NUM_OF_LINEAR_CELLS = Config.num_of_linear_cell
NUM_OF_TAEGET_PC = 256
STANDARD_DEVIATION_PC = 0.01
NUM_OF_TAEGET_HDC = 12
CONCENTRATE_PAR_HDC = 20.


class RatTrajectoryModel(nn.Module):
    """
    input(v, sin(theta), cons(theta))
    LSTM:
    The initial cell state and hidden state of the LSTM
    l_0 = W^cp c_0 + W^cd h_0
    m_0 = W^hp c_0 + W^hd h_0
    Linear:
    512 neurous
    dropout (0.5)

    """

    def __init__(self, input_size: int, seq_len: int) -> None:
        super().__init__()
        self.recurrent_layer = nn.LSTM(
            input_size=input_size, hidden_size=LSTM_HIDDEN_UNITS,
            num_layers=NUM_OF_LSTM_HIDDEN_LAYERS, batch_first=True)
        self.linear_layer = nn.Linear(
            LSTM_HIDDEN_UNITS, NUM_OF_LINEAR_CELLS, bias=False)
        self.dropout = nn.Dropout(p=Config.dropout_rate)
        self.output_layer_pc = nn.Linear(NUM_OF_LINEAR_CELLS, NUM_OF_TAEGET_PC)
        self.output_layer_hdc = nn.Linear(
            NUM_OF_LINEAR_CELLS, NUM_OF_TAEGET_HDC)

        self.Wcp = nn.Linear(NUM_OF_TAEGET_PC, LSTM_HIDDEN_UNITS, bias=False)
        self.Wcd = nn.Linear(NUM_OF_TAEGET_HDC, LSTM_HIDDEN_UNITS, bias=False)
        self.Whp = nn.Linear(NUM_OF_TAEGET_PC, LSTM_HIDDEN_UNITS, bias=False)
        self.Whd = nn.Linear(NUM_OF_TAEGET_HDC, LSTM_HIDDEN_UNITS, bias=False)

        # self.bn_pc = torch.nn.BatchNorm1d()
    def forward(self, x, x0):
        """
        x.shape: [N, L, H]
        N: batch size.
        L: length of seq.
        H: feature flatten shape
        x0 (c0, h0)
        c0: [batch size, num of target place cell]
        h0: [batch size, num of target head direction cell]
        """
        c0 = x0[0]
        h0 = x0[1]
        # 线性映射到lstm的cell state初始化 [batch size, hidden size]
        l0 = self.Wcp(c0) + self.Wcd(h0)
        # 线性映射到lstm的hidden layer初始化 [batch size, hidden size]
        m0 = self.Whp(c0) + self.Whd(h0)
        m0 = m0.permute(1, 0, 2)
        l0 = l0.permute(1, 0, 2)
        # input [batch size, seq len, input size] h [D∗num_layers, batch size, hidden size]
        h, _ = self.recurrent_layer(x, (m0, l0))
        g = self.dropout(self.linear_layer(h))
        pre_pc = self.output_layer_pc(g)
        pre_hdc = self.output_layer_hdc(g)
        return F.softmax(pre_pc, dim=-1), F.softmax(pre_hdc, dim=-1)


# %%
if __name__ == '__main__':
    model = RatTrajectoryModel(22, 120)
    weight_decay_para_keys = ['linear_layer.weight', 'output_layer_pc.weight', 'output_layer_hdc.weight']
    # parameters = dict(model.named_parameters())
    # weight_decay_paras = {key:parameters[key] for key in weight_decay_para_keys}
    weight_decay_list = (param for name, param in model.named_parameters() if name in weight_decay_para_keys)
    no_decay_list = (param for name, param in model.named_parameters() if name not in weight_decay_para_keys)

    x0 = (torch.randn(3, 1, NUM_OF_TAEGET_PC),
          torch.randn(3, 1, NUM_OF_TAEGET_HDC))
    x = torch.randn(3, 120, 22)
    pre = model(x, x0)
