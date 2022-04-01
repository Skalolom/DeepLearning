import torch
import torch.nn as nn
from torch import Tensor
import math


class LSTMCell(nn.Module):

    """
    Implementation of the LSTM cell
    """

    def __init__(self, input_size: int, hidden_size: int, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x_h = nn.Linear(input_size, 4 * hidden_size, bias=bias)        # linear transformation 
        self.h_prev_h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)       # linear transformation
        self.c_prev_c = Tensor(hidden_size * 3)
        self.reset_parameters()



    def reset_parameters(self):
        # reset weight matrices
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):

        # hidden state on previous step and input tensors
        h_x, c_x = hidden
        
        # reshape x
        x = x.view(-1, x.size(1))
        
        # calc linear transformations
        gates = self.x_h(x) + self.h_prev_h(h_x)
        
        # reshape gates and 
        gates = gates.squeeze()
        
        c_prev_c = self.c_prev_c.unsqueeze(0)

        # split tensors
        c_i, c_f, c_o = c_prev_c.chunk(3,1)
        i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)
        
        # apply non-linear transformations
        i_gate = torch.sigmoid(i_gate + c_i * c_x)
        f_gate = torch.sigmoid(f_gate + c_f * c_x)
        c_gate = f_gate * c_x + i_gate * torch.tanh(c_gate)
        o_gate = torch.sigmoid(o_gate + c_o * c_gate)
        

        # calc current h
        h_m = o_gate * torch.tanh(c_gate)

        return (h_m, c_gate)