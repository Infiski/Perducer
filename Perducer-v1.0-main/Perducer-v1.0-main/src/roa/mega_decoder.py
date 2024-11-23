import torch
import torch.nn as nn
import torch.nn.functional as F

class MEGADecoder(nn.Module):
    def __init__(self, r_dim, r_size, p_dim, at_dim, o_dim):
        super(MEGADecoder, self).__init__()
        self.r_dim = r_dim
        self.r_size = r_size
        self.p_dim = p_dim
        self.at_dim = at_dim
        self.o_dim = o_dim

        self.W_alpha = nn.Parameter(torch.Tensor(r_dim, r_dim * 2))
        self.b_alpha = nn.Parameter(torch.Tensor(r_dim))

        self.W_delta = nn.Parameter(torch.Tensor(r_dim, r_dim * 2))
        self.b_delta = nn.Parameter(torch.Tensor(r_dim))

        self.W_EMA = nn.Parameter(torch.Tensor(p_dim, r_dim))
        self.b_EMA = nn.Parameter(torch.Tensor(p_dim))
    
        self.W_q = nn.Parameter(torch.Tensor(at_dim, p_dim))
        self.b_q = nn.Parameter(torch.Tensor(at_dim))
    
        self.W_k = nn.Parameter(torch.Tensor(at_dim, p_dim))
        self.b_k = nn.Parameter(torch.Tensor(at_dim))
    
        self.W_v = nn.Parameter(torch.Tensor(at_dim, p_dim))
        self.b_v = nn.Parameter(torch.Tensor(at_dim))

        self.W_f = nn.Parameter(torch.Tensor(at_dim, p_dim))
        self.b_f = nn.Parameter(torch.Tensor(at_dim))

        self.W_EMA_c = nn.Parameter(torch.Tensor(p_dim, r_dim))
        self.W_z_C = nn.Parameter(torch.Tensor(at_dim, r_dim))
        
        self.b_C = nn.Parameter(torch.Tensor(r_dim))

        self.W_i = nn.Parameter(torch.Tensor(r_dim, p_dim))
        self.b_i = nn.Parameter(torch.Tensor(r_dim))

        self.W_o = nn.Parameter(torch.Tensor(o_dim, r_dim))
        self.b_o = nn.Parameter(torch.Tensor(o_dim))

        
    
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_alpha)
        nn.init.xavier_uniform_(self.W_delta)
        nn.init.xavier_uniform_(self.W_EMA)
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)
        nn.init.xavier_uniform_(self.W_f)
        nn.init.xavier_uniform_(self.W_EMA_c)
        nn.init.xavier_uniform_(self.W_z_C)
        nn.init.xavier_uniform_(self.W_i)
        nn.init.xavier_uniform_(self.W_o)

        nn.init.zeros_(self.b_alpha)
        nn.init.zeros_(self.b_delta)
        nn.init.zeros_(self.b_EMA)
        nn.init.zeros_(self.b_q)
        nn.init.zeros_(self.b_k)
        nn.init.zeros_(self.b_v)
        nn.init.zeros_(self.b_f)
        nn.init.zeros_(self.b_C)
        nn.init.zeros_(self.b_i)
        nn.init.zeros_(self.b_o)
    def forward(self, R):
        R_1 = torch.cat((torch.zeros(1, self.r_dim), R), dim=0)
        m_size, _ = R.shape # m_size : size of the modified R (r_size + 1)

        r_t = R_1[1:]
        r_t_prev = R_1[:-1]

        concat_r = torch.cat((r_t_prev, r_t), dim=1)
        alpha_t = torch.tanh(F.linear(concat_r, self.W_alpha, self.b_alpha))  # Compute alpha_t
        delta_t = torch.tanh(F.linear(concat_r, self.W_delta, self.b_delta))  # Compute delta_t

        R_EMA = alpha_t * r_t + (1 - alpha_t) * delta_t * r_t_prev

        silu = nn.SiLU()

        R_EMA_prime = F.linear(R_EMA, self.W_EMA, self.b_EMA)
        R_EMA_prime = silu(R_EMA_prime)
        
        Q = F.linear(R_EMA_prime, self.W_q, self.b_q)
        K = F.linear(R_EMA_prime, self.W_k, self.b_k)
        V = F.linear(R_EMA_prime, self.W_v, self.b_v)

        at_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.at_dim))
        at_weights = F.softmax(at_scores, dim=-1)
        
        Z_EMA = torch.matmul(at_weights, V)

        f = torch.sigmoid(F.linear(R_EMA_prime, self.W_f, self.b_f))
        
        Z_EMA_f = f * Z_EMA

        Z_EMA_C = torch.matmul(R_EMA_prime, self.W_EMA_c) + torch.matmul(Z_EMA_f, self.W_z_C) + self.b_C
        Z_EMA_C = silu(Z_EMA_C)

        i = torch.sigmoid(F.linear(R_EMA_prime, self.W_i, self.b_i))
        R_h = i * Z_EMA_C + (1 - i) * R
        
        r_cap = F.linear(R_h, self.W_o, self.b_o)
        p_cap = torch.sigmoid(r_cap)

        return p_cap