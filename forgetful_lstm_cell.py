import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import math
import pdb


class ForgetfulGRUCell(nn.Module):
  """Similar to GRU cell, but allows dropout during inference

  Definition:
    r = sigma(x_t U_r + h_{t-1} W_r + b_r)
    z = sigma(x_t U_z + h_{t-1} W_z + b_z)
    hbar = tanh(x_t U_hbar + (r*h_{t-1}) W_hbar + b_hbar)
    h_t = dropout(z*h_{t-1} + (1-z)*hbar)

  Notation:
    x_t: input
    h_{t-1}: previous hidden state
    h_t: next hidden state
    r: reset gate
    z: update gate
    U: weights operating on x_t
    W: weights operating on h_{t-1}
    b: bias
    hbar: temporary hidden state
    sigma: activation function with range [0, 1]
    tanh: activation function with range [-1, 1]
  """

  def __init__(self, input_size, hidden_size):
    super(ForgetfulGRUCell, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.U_r = Parameter(torch.Tensor(input_size, hidden_size))
    self.W_r = Parameter(torch.Tensor(hidden_size, hidden_size))
    self.b_r = Parameter(torch.Tensor(hidden_size))
    self.U_z = Parameter(torch.Tensor(input_size, hidden_size))
    self.W_z = Parameter(torch.Tensor(hidden_size, hidden_size))
    self.b_z = Parameter(torch.Tensor(hidden_size))
    self.U_hbar = Parameter(torch.Tensor(input_size, hidden_size))
    self.W_hbar = Parameter(torch.Tensor(hidden_size, hidden_size))
    self.b_hbar = Parameter(torch.Tensor(hidden_size))
    self.reset_parameters()

  def reset_parameters(self):
    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)

  def forward(self, x, h):
    r = F.sigmoid(x.mm(self.U_r) + h.mm(self.W_r) + self.b_r)
    z = F.sigmoid(x.mm(self.U_z) + h.mm(self.W_z) + self.b_z)
    hbar = F.tanh(x.mm(self.U_hbar) + (r * h).mm(self.W_hbar) + self.b_hbar)
    hnext = z * h + (1-z) * hbar
    # Todo: add dropout here
    return hnext
