import torch
import torch.nn as nn
import torch.nn.functional as F

from .b_cell import BCell

class BTier(nn.Module):
    def __init__(self, i_dim, h_dim, b_dim):
        super(BTier, self).__init__()
        self.i_dim = i_dim
        self.h_dim = h_dim
        self.b_dim = b_dim

        self.b_cell = BCell(i_dim, h_dim, b_dim)
        
    def forward(self, x, hidden=None):
        nh_batch, nr_batch, nt_batch, _ = x
        batch_size, seq_len, _ = nh_batch.shape

        if hidden is None:
            h = torch.zeros(batch_size, self.h_dim, device=nh_batch.device)
        else:
            h = hidden

        outputs = []
        for t in range(seq_len):
            nh_t = nh_batch[:, t, :]
            nr_t = nr_batch[:, t, :]
            nt_t = nt_batch[:, t, :]
            
            h, b = self.b_cell(h, nh_t, nr_t, nt_t)
            outputs.append(b)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, h