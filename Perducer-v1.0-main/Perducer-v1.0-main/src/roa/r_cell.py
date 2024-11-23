import torch
import torch.nn as nn
import torch.nn.functional as F

class MEGADecoder(nn.Module):
    def __init__(self, r_dim, r_size, z_dim, at_dim, f_dim, g_dim):
        super(MEGADecoder, self).__init__()
        self.r_dim = r_dim
        self.r_size = r_size
        self.z_dim = z_dim
        self.at_dim = at_dim
        self.g_dim = g_dim
        self.f_dim = f_dim

        self.W_alpha = nn.Parameter(torch.Tensor(r_dim, r_dim * 2))
        self.b_alpha = nn.Parameter(torch.Tensor(r_dim))

        self.W_delta = nn.Parameter(torch.Tensor(r_dim, r_dim * 2))
        self.b_delta = nn.Parameter(torch.Tensor(r_dim))

        self.W_q = nn.Parameter(torch.Tensor(at_dim, z_dim))
        self.b_q = nn.Parameter(torch.Tensor(at_dim))
    
        self.W_k = nn.Parameter(torch.Tensor(at_dim, z_dim))
        self.b_k = nn.Parameter(torch.Tensor(at_dim))
    
        self.W_v = nn.Parameter(torch.Tensor(at_dim, z_dim))
        self.b_v = nn.Parameter(torch.Tensor(at_dim))
        
        self.W_z = nn.Parameter(torch.Tensor(z_dim, r_dim))
        self.b_z = nn.Parameter(torch.Tensor(z_dim))
    

    
        self.W_f = nn.Parameter(torch.Tensor(at_dim, g_dim))
        self.b_f = nn.Parameter(torch.Tensor(at_dim))
    
        self.W_EMA = nn.Parameter(torch.Tensor(g_dim, r_dim))
        self.b_EMA = nn.Parameter(torch.Tensor(g_dim))
    
        self.W_z_at = nn.Parameter(torch.Tensor(g_dim, at_dim))
        self.b_z_at = nn.Parameter(torch.Tensor(g_dim))

        
    
        self.W_i = nn.Parameter(torch.Tensor(1, r_dim))
        self.b_i = nn.Parameter(torch.Tensor(1))

        
    
        self.W_final = nn.Parameter(torch.Tensor(1, g_dim))
    
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weight parameters with Xavier uniform initialization
        nn.init.xavier_uniform_(self.W_alpha)
        nn.init.xavier_uniform_(self.W_z)
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)
        nn.init.xavier_uniform_(self.W_f)
        nn.init.xavier_uniform_(self.W_EMA)
        nn.init.xavier_uniform_(self.W_z_at)
        nn.init.xavier_uniform_(self.W_i)
        nn.init.xavier_uniform_(self.W_final)
        
        # Initialize bias parameters to zeros
        nn.init.zeros_(self.b_alpha)
        nn.init.zeros_(self.b_z)
        nn.init.zeros_(self.b_q)
        nn.init.zeros_(self.b_k)
        nn.init.zeros_(self.b_v)
        nn.init.zeros_(self.b_f)
        nn.init.zeros_(self.b_EMA)
        nn.init.zeros_(self.b_z_at)
        nn.init.zeros_(self.b_i)

    def forward(self, R):
        R = torch.cat((torch.zeros(1, self.r_dim), R), dim=0)
        m_size, _ = R.shape # m_size : size of the modified R (r_size + 1)

        r_t = R[1:]
        r_t_prev = R[:-1]

        concat_r = torch.cat((r_t_prev, r_t), dim=1)
        
        alpha_t = torch.tanh(F.linear(concat_r, self.W_alpha, self.b_alpha))  # Compute alpha_t
        delta_t = torch.tanh(F.linear(concat_r, self.W_delta, self.b_delta))  # Compute delta_t

        # R_EMA
        R_EMA = alpha_t * r_t + (1 - alpha_t) * delta_t * r_t_prev
        silu = nn.SiLU()

        Z = F.linear(R_EMA, self.W_z, self.b_z)
        Z = silu(Z)
        
        Q = F.linear(Z, self.W_q, self.b_q)
        K = F.linear(Z, self.W_k, self.b_k)
        V = F.linear(Z, self.W_v, self.b_v)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.at_dim))
        attention_weights = F.softmax(attention_scores, dim=-1)
        Z_at = torch.matmul(attention_weights, V)

        

        R_EMA_prime = F.linear(R_EMA, self.W_EMA, self.b_EMA)
        f = torch.sigmoid(F.linear(R_EMA_prime, self.W_f, self.b_f))
        Z_at_prime = f * Z_at
        
        Z_EMA_f = F.linear(Z_at_prime, self.W_z_at, self.b_z_at)
        

        Z_triple_prime = torch.tanh(R_EMA_prime + Z_EMA_f)

        i = torch.tanh(F.linear(R_EMA, self.W_i, self.b_i))

        Z_final = i * Z_triple_prime + (1 - i) * R_EMA_prime

        p_hat = torch.sigmoid(F.linear(Z_final, self.W_final))
        return p_hat