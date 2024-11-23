import torch
import torch.nn as nn
import torch.nn.functional as F

class MEGADecoder(nn.Module):
    def __init__(self, b_size, b_dim, o_dim):
        super(MEGADecoder, self).__init__()
        self.b_dim = b_dim
        self.b_size = b_size
        self.o_dim = o_dim

        self.ema = EMA(b_size=b_size, b_dim=self.b_dim)
        
        self.attention = Attention(b_dim=self.b_dim)

        self.W_f = nn.Parameter(torch.Tensor(b_dim, b_dim))
        self.b_f = nn.Parameter(torch.Tensor(b_dim))

        self.W_EMA_c = nn.Parameter(torch.Tensor(b_dim, b_dim))
        self.W_z_C = nn.Parameter(torch.Tensor(b_dim, b_dim))
        
        self.b_C = nn.Parameter(torch.Tensor(b_dim))

        self.W_i = nn.Parameter(torch.Tensor(b_dim, b_dim))
        self.b_i = nn.Parameter(torch.Tensor(b_dim))

        self.W_o = nn.Parameter(torch.Tensor(o_dim, b_dim))
        self.b_o = nn.Parameter(torch.Tensor(o_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_f)
        nn.init.xavier_uniform_(self.W_EMA_c)
        nn.init.xavier_uniform_(self.W_z_C)
        nn.init.xavier_uniform_(self.W_i)
        nn.init.xavier_uniform_(self.W_o)
        
        nn.init.zeros_(self.b_f)
        nn.init.zeros_(self.b_C)
        nn.init.zeros_(self.b_i)
        nn.init.zeros_(self.b_o)

    # expected dimension of B : (batch, b_size, b_dim)
    def forward(self, b, b_mask=None):
        b_ema = self.ema(b)

        # print("DEBUG _ B_EMA", b_ema)
        
        Z_EMA, _ = self.attention(b_ema, b_mask)

        f = torch.sigmoid(F.linear(b_ema, self.W_f, self.b_f))
        
        Z_EMA_f = f * Z_EMA

        Z_EMA_C = torch.matmul(b_ema, self.W_EMA_c) + torch.matmul(Z_EMA_f, self.W_z_C) + self.b_C

        Z_EMA_C = F.silu(Z_EMA_C)

        i = torch.sigmoid(F.linear(b_ema, self.W_i, self.b_i))
        B_h = i * Z_EMA_C + (1 - i) * b
        
        b_cap = F.linear(B_h, self.W_o, self.b_o)
        p_cap = torch.sigmoid(b_cap)

        return p_cap

class EMA(nn.Module):
    def __init__(self, b_size, b_dim):
        super(EMA, self).__init__()
        self.b_size = b_size
        self.b_dim = b_dim

        self.alpha = nn.Linear(b_dim * 2, b_dim)
        self.delta = nn.Linear(b_dim * 2, b_dim)

        self.output = nn.Linear(b_dim, b_dim)

    # expected dimension of b : (batch, b_size, b_dim)
    def forward(self, b):
        batch, _, _ = b.size()
        b_t_ema_seq = []
        b_t_ema_prev = torch.zeros(batch, self.b_dim)
        
        # (b_size, b_dim)
        # for b_t in b:
        #     con = torch.cat((b_t_ema_prev, b_t.unsqueeze(0)), dim = 1)
        #     alpha_t = self.alpha(con)
        #     delta_t = self.delta(con)
        #     b_t_ema = alpha_t * b_t + (1 - alpha_t * delta_t) * b_t_ema_prev
        #     b_t_ema_seq.append(b_t_ema.squeeze(0))
        #     b_t_ema_prev = b_t_ema
        # b_ema_seq = torch.stack(b_t_ema_seq)

        # output = self.output(b_ema_seq)

        # return F.silu(output)

        for t in range(self.b_size):
            b_t = b[:, t, :]  # Get the t-th time step for all sequences
            con = torch.cat((b_t_ema_prev, b_t), dim=1)  # Concatenate previous EMA with current input
            alpha_t = torch.tanh(self.alpha(con))
            delta_t = torch.tanh(self.delta(con))

            # `sigmoid`` is used here to stabilize the output, avoiding NaNs caused by extreme values in long sequences 
            # due to the cumulative effect of the Hadamard product.
            b_t_ema = torch.sigmoid(alpha_t * b_t + (1 - alpha_t * delta_t) * b_t_ema_prev)
            b_t_ema_seq.append(b_t_ema)
            b_t_ema_prev = b_t_ema

        b_ema_seq = torch.stack(b_t_ema_seq, dim=1)  # Shape: (batch_size, seq_len, b_dim)
        
        output = self.output(b_ema_seq)

        return F.silu(output)
            
class Attention(nn.Module):
    def __init__(self, b_dim):
        super(Attention, self).__init__()
        self.query = nn.Linear(b_dim, b_dim)
        self.key = nn.Linear(b_dim, b_dim)
        self.value = nn.Linear(b_dim, b_dim)
        self.scale = torch.sqrt(torch.FloatTensor([b_dim]))

    def forward(self, b, mask=None):
        q = self.query(b)
        k = self.key(b)
        v = self.value(b)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights




