import torch
import torch.nn as nn
import torch.nn.functional as F

class B_Cell(nn.Module):
    # i_dim = input dimension
    # h_dim = hidden dimension
    # b_dim = behavior dimension
    def __init__(self, i_dim, h_dim, b_dim):
        super(B_Cell, self).__init__()
        self.i_dim = i_dim
        self.h_dim = h_dim
        self.b_dim = b_dim
        
        self.W_nh = nn.Parameter(torch.Tensor(h_dim, i_dim))
        self.b_nh = nn.Parameter(torch.Tensor(h_dim))
        
        self.W_nr = nn.Parameter(torch.Tensor(h_dim, i_dim))
        self.b_nr = nn.Parameter(torch.Tensor(h_dim))
        
        self.W_nt = nn.Parameter(torch.Tensor(h_dim, i_dim))
        self.b_nt = nn.Parameter(torch.Tensor(h_dim))

        self.W_bt = nn.Parameter(torch.Tensor(b_dim, h_dim))
        self.b_bt = nn.Parameter(torch.Tensor(b_dim))

        self.W_h = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.b_h = nn.Parameter(torch.Tensor(h_dim))

        self.init_weights()


    def init_weights(self):
        nn.init.xavier_uniform_(self.W_nh)
        nn.init.xavier_uniform_(self.W_nr)
        nn.init.xavier_uniform_(self.W_nt)
        nn.init.xavier_uniform_(self.W_bt)
        nn.init.xavier_uniform_(self.W_h)
        nn.init.zeros_(self.b_nh)
        nn.init.zeros_(self.b_nr)
        nn.init.zeros_(self.b_nt)
        nn.init.zeros_(self.b_bt)
        nn.init.zeros_(self.b_h)

    def forward(self, h_prev, n_h, n_r, n_t):
        # ref: https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html

        # nh Cell
        h_i = torch.tanh(F.linear(n_h, self.W_nh, self.b_nh)) + torch.tanh(F.linear(h_prev, self.W_h, self.b_h))
        
        # nr Cell
        # Projection
        temp = (n_r)/torch.norm(n_r)
        h_i_p = h_i - torch.mm(torch.mm(temp.t(), h_i).t(), temp.t()).t()
        
        h_i_1 = torch.tanh(F.linear(n_r, self.W_nr, self.b_nr)) + torch.tanh(F.linear(h_i_p, self.W_h, self.b_h))
        
        # nt Cell
        # Projection
        temp = (n_t)/torch.norm(n_t)
        h_i_p = h_i_1 - torch.mm(torch.mm(temp.t(), h_i_1).t(), temp.t()).t()
        
        h_next = torch.tanh(F.linear(n_t, self.W_nt, self.b_nt)) + torch.tanh(F.linear(h_i_p, self.W_h, self.b_h))

        # Output
        b = torch.tanh(F.linear(h_next, self.W_bt, self.b_bt))
        
        return h_next, b

class OneHotPositionalEncoding(nn.Module):

    def __init__(self, b_dim: int, seq_len: int) -> None:
        super().__init__()
        self.b_dim = b_dim
        self.seq_len = seq_len 

        one_hot_positional_encoding = torch.eye(seq_len, b_dim)

        one_hot_positional_encoding = one_hot_positional_encoding.unsqueeze(0) # (1, seq_len, b_dim)

        self.register_buffer('one_hot_positional_encoding', one_hot_positional_encoding)

    def forward(self, x):
        x = x + (self.one_hot_positional_encoding[:, :x.shape[1], :]).required_grad_(False)
        return x