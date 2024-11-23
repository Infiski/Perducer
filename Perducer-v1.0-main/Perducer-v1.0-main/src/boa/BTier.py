import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import BCell

class B_Tier(nn.Module):
    def __init__(self, i_dim, h_dim, b_dim):
        super(B_Tier, self).__init__()
        self.i_dim = i_dim
        self.h_dim = h_dim
        self.b_dim = b_dim

        self.b_cell = BCell.B_Cell(self.i_dim, self.h_dim, self.b_dim)

    # expected dimension of x : (batch, 3 (head, relation, tail), seq_len, i_dim)
    def forward(self, x, hidden=None):

        # unpack the input tensor x into nh_batch, nr_batch, and nt_batch
        nh_batch, nr_batch, nt_batch = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :]

        # Get the batch size and sequence length
        batch_size, seq_len, _ = nh_batch.shape

        # Initialize hidden state if not provided
        if hidden is None:
            h = torch.zeros(batch_size, self.h_dim, device=nh_batch.device)
        else:
            h = hidden

        # Pre-allocate the output tensor
        outputs = torch.empty(batch_size, seq_len, self.b_dim, device=x.device)

        # Iterate over the sequence length
        for t in range(seq_len):
            # Extract the t-th time step for each component
            nh_t = nh_batch[:, t, :]  # Shape: (batch_size, d_model)
            nr_t = nr_batch[:, t, :]  # Shape: (batch_size, d_model)
            nt_t = nt_batch[:, t, :]  # Shape: (batch_size, d_model)
            
            # Apply your custom cell (e.g., self.b_cell) to update the hidden state and get the output
            h, b = self.b_cell(h, nh_t, nr_t, nt_t)
            outputs[:, t, :] = b  # Directly assign the output to the pre-allocated tensor
        
        return outputs, h

