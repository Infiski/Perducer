import torch
import torch.nn as nn
import torch.nn.functional as F

from .r_cell import RCell

class RTier(nn.Module):
    def __init__(self, b_dim, h_dim, r_dim):
        super(RTier, self).__init__()
        self.b_dim = b_dim
        self.h_dim = h_dim
        self.r_dim = r_dim

        self.r_cell = RCell(b_dim, h_dim, r_dim)
        
    def forward(self, x, hidden=None):
        b_batch, is_summ, _ = x
        batch_size, seq_len, _ = b_batch.shape

        if hidden is None:
            h = torch.zeros(batch_size, self.h_dim, device=b_batch.device)
        else:
            h = hidden

        outputs = []
        for t in range(seq_len):
            b = b_batch[:, t, :]
            h, r = self.r_cell(h, b, is_summ[:, t])
            outputs.append(r)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, h