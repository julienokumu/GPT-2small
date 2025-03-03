import torch
import torch.nn as nn
from datasets import load_dataset

GPT2_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 64,
    "emb_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out & num_heads == 0), "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # size per head
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)) # casual attention

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        # reshape for multiheads
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        # swap with tokens
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # calc attn scores
        attn_scores = queries @ keys.transpose(2, 3)
        # prevent future peeking
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # set future tokens to -infinity so they're ignored
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # normalize scores, stability
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # combine w + values = attn out
        context_vec = (attn_weights @ values).transpose(1, 2)
        # flatten back to original shape
        context_vec = context_vec.contagious().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec # return attn-processed data
