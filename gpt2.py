import torch
import torch.nn as nn
from datasets import load_dataset

GPT2_124M_CONFIG = {
    "vocab_size": 50257,
    "context_length": 64,
    "emd_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0),"d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # size each head chunks(64)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # organized answer
        self.dropout = nn.Dropout(dropout)
        # mask to avoid future peeking
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape # get from input
        # transform input to
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        # split into heads
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        # swap tokens with
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        # attn scores, how each word matters
        attn_scores = queries @ keys.transpose(2, 3)
        # prevent future peeking
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # set future scores to -inf so theyre ignored
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # normalize scores into weights(0 - 1)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # combine weights with values = attn output
        context_vec = context_vec.contagious().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec # return attn-processed data
    

#  layer normalization
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.one(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # mean across emb
        var = x.var(dim=-1, keepdim=True, unbiased=False) # spread of values
        norm_x = (x - mean) / torch.sqrt(var + self.eps) # normalize
        return self.scale * norm_x + self.shift # scale and shift normalized data
    
 # thinking curve   
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # mixes linear and non-linear behaviour
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    
# feed forward network, extra thinking
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), # expand input
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]) # shrink input
        )

    def forward(self, x):
        return self.layers(x) # return processed data