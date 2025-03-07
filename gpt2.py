import torch
import torch.nn as nn
from datasets import load_dataset

"""
vocab_size = number of tokens our model can handle
context_length = number of tokens our model can look at at once
emb_dim = vector representation of each word
n_heads = number of attention heads
n_layers = number of transformer layers
qkv_bias = whether to add bias to q, k, v
drop_rate = chances of ignoring some data to avoid overfitting
attn_scores = how much each word cares about other words
tokens = individual words
logits = predictions scores for each word likely to come next
transformer = layer that combine attention and feedforward network
query = questions to ask
key = what to match against
value = answers
"""

GPT2_124M_CONFIG = {
    "vocab_size": 50257,
    "context_length": 64,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "qkv_bias": False,
    "drop_rate": 0.1
}

# multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # chunk per head

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out) # final layer combining attn outputs
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length)))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # turn input to
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        # reshape for attn heads
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        # swap dims
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        # mask to prevent future peeking
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # future tokens to -inf so theyre ignored
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # normalize with softmax to turn scores to weights
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # weighted sum of values
        context_vec = (attn_weights @ values).transpose(1, 2)
        # flatten back to original shape
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # combine all heads
        return context_vec

# layer norm
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 # avoid division by zero
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # mean across emb
        var = x.var(dim=-1, keepdim=True) # spread of values
        norm_x = (x - mean) / torch.sqrt(var - self.eps)
        return self.scale * norm_x * self.shift

# thinking curve
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

# feedforward network
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), # expand
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]) # shrink
        )

    def forward(self, x):
        return self.layers(x)

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
        # feedforward with config settings
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x # save input for residual connection
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut # residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x # processed data

# gpt2 model
class GPT2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # token and positional emb layers
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # 12 transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False) # predict next token

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        # add positional info to each token
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x) # prediction scores for next tokens
        return logits

# tokenization
def simple_tokenizer(text, max_vocab=1000):
    words = text.split()
    word_to_idx = {word: idx for idx, word in enumerate(set(words[:max_vocab]))} # map words to numbers
    tokens = [word_to_idx.get(word, 0) for word in words] # each word to a number and 0 if unk
    return torch.tensor(tokens, dtype=torch.long) # turn list to tensor of numbers

def simple_detokenizer(tokens, text, max_vocab=1000):
    words = text.split()
    word_to_idx = {word: idx for idx, word in enumerate(set(words[:max_vocab]))}
    # number to words
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return [idx_to_word.get(t.item(), "<unk>") for t in tokens]

# prepare dataset
def prepare_data():
    with open("tiny_shakespeare.txt", "r") as f:
        text = f.read()
    return text

# split tokens into batches
def create_batches(text, batch_size=1):
    # text to a list of tokens
    tokens = simple_tokenizer(text)
    seq_len = GPT2_124M_CONFIG["context_length"]
    num_tokens = len(tokens)
    num_batches = max(1, num_tokens // seq_len)
    total_length = num_batches * seq_len # tokens needed for full batch
    # if too short pad with zeros to reach total_length
    if num_tokens < total_length:
        tokens = torch.cat([tokens, torch.zeros(total_length - num_tokens, dtype=torch.long)])
    tokens = tokens[:total_length] # trim to exact length
    tokens = tokens.view(batch_size, num_batches, seq_len)
    inputs = tokens[:, :, :-1] # take all but last token
    targets = tokens[:, :, 1:] # take all but first token as targets
    return inputs, targets

# training
def train_gpt(model, text, num_epochs=1, batch_size=1, learning_rate=1e-3):
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    inputs, targets = create_batches(text, batch_size)
    num_batches = inputs.size(1)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx in range(num_batches):
            input_batch = inputs[:, batch_idx, :].to(device)
            target_batch = targets[:, batch_idx, :].to(device)
            optimizer.zero_grad()
            logits = model(input_batch) # run model to get prediction
            # loss between predictions and targets
            loss = criterion(logits.view(-1, GPT2_124M_CONFIG["vocab_size"]), target_batch.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # limiti tweaks to avoid explosions
            optimizer.step()
            total_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}")
        avg_loss = total_loss / num_batches
        print(f"Epoch: {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    return model

# text generation
def generate(model, input_text, text, max_len=20):
    model.eval()
    tokens = simple_tokenizer(input_text)
    input_ids = tokens.unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(input_ids) # get predictions for next token
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1) # add to sequence
        output_tokens = input_ids[0] # final token list
    # tokens back to words and join into a string
    return " ".join(simple_detokenizer(output_tokens, text))

if __name__ == "__main__":
    model = GPT2Model(GPT2_124M_CONFIG)
    text = prepare_data()
    trained_model = train_gpt(model, text, num_epochs=1, batch_size=1)
    torch.save(trained_model.state_dict(), "gpt-2small_124m_cpu.pth")
    input_text = "To be or not to be"
    generated_text = generate(trained_model, input_text, text)
    print("GPT-2small: ", generated_text)


