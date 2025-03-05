import torch
import torch.nn as nn
from datasets import load_dataset

"""
vocab_size = total number of tokens our model can handle
context_length = total numner of tokens our model looks at at once
emb_dim = vector representation of each word
n_head = number of attention heads
n_layer = number of transformer layers
qkv_bias = whether to include a bias in qkv
drop_rate = rate at which our model ignores some data to avoid overfitting
tokens = individual words
logits = prediction scores for which word is likely to come next
transformer = layer that combines attention and feedforwardnet
query = questions to ask
key = what we are looking for
value = what anser to give
attn_scores = how much each word cares about other words
"""

GPT2_124M_CONFIG = {
    "vocab_size": 50257,
    "context_length": 64,
    "emb_dim": 768,
    "n_head": 12,
    "n_layer": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # chunk per head
        # linear layers for the q, k, v
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # final layer combining attn outputs
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        # prevent future peeking
        self.register_buffer = ("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    # data flow in mah
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # transform input to q, k, v
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        # reshape for attn heads
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        # swap tokens with
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # attn scores
        attn_scores = queries @ keys.transpose(2, 3)
        # mask out future peeking
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # future score to -infinitiy so theyre ignored
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # normalize scores into weight, softmax
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # combine weights to values
        context_vec = (attn_weights @ values).transpose(1, 2)
        # flatten back to original shape
        context_vec = context_vec.contaguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # final linear transformation
        return context_vec # return attn-processed data
    
# layer normalization
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # mean across last emb
        var = x.var(dim=-1, keepdim=True, unbiased=False) # spread of values
        norm_x = (x - mean) / torch.sqrt(var + self.eps) # normalize
        return self.scale * norm_x + self.shift # scale and shift the normalized data
    
# thinking curve
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # mixes linear and non-linear behaviour
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    
# feedforward network
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # sequence of layers
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), # expand input
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), # shrink back
        )

    def forward(self, x):
        return self.layers(x) # pass input through layers and return result
    
# transformer block, attn + feedforward
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # attn with config settings
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        # feedforward net with config settings
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x # save input for residual connection
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut # add original input
        shorcut = x # save input for next residual connection
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x # return processed data
    
# GPT2 Model
class GPT2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # token and positional emb layers
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # stack 12 transformer blocks sequentially
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False) # output layer to predict tokens

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx) # convert token ids to emb
        # positional emb for each position in the sequence
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = pos_embeds + tok_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x) # pass through all trf blocks
        x = self.final_norm(x)
        logits = self.out_head(x) # convert to logits
        return logits # return the predictions
        
