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
        # linear layers for q, k, v
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # final layer that combines  attn outputs
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        # mask to prevent future peeking
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # transform input to
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        # reshape for attn heads
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        # swap token with
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        # attn scores
        attn_scores = queries @ keys.transpose(2, 3)
        # mask matching current token count
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # set future tokens to -inf so theyre ignored
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # apply softmax
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # combine weights with values to get attn output
        context_vec = (attn_weights @ values).transpose(1, 2)
        # flatten back to original shape
        context_vec = context_vec.contagious().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec # attn processed data
    
# layer normalization
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # mean across last emb
        var = x.var(dim=-1, keepdim=True, unbiased=False) # spread of values across emb
        norm_x = (x - mean) / torch.sqrt(var + self.eps) # normalize
        return self.scale * norm_x + self.shift # scale and shift normalized data
    
# thinking curve
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    
# feedforward network
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # sequential layers
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), # expand input
            GELU(),
            nn.Linear( 4 * cfg["emb_dim"], cfg["emb_dim"]) # shrink back input
        )

    def forward(self, x):
        return self.layers(x) # pass input through the layers and return result
    
# transformer block
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
        # feedforward network
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x # save input for residual connection
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut # add original input(residual connection)
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x # return processed data

# gpt model
class GPT2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # token and positional emb layers
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # stack 12 transformer layers
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False) # output layer to predict tokens

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx) # token ids to emb
        # positional emb for each position in the sequence
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x) # pass through all trf layers
        x = self.final_norm(x)
        logits = self.out_head(x) # convert to logits     
        return logits # return the predictions
    
# tokenization
def simple_tokenizer(text, max_vocab=1000):
    words = text.split()
    # map words to numbers
    word_to_idx = {word: idx for idx, word in enumerate(set(words[:max_vocab]))}
    # map words to tokens
    tokens = [word_to_idx(word, 0) for word in words]
    seq_len = GPT2_124M_CONFIG["context_length"] # max seq length
    # pas with zeros if too short
    if len(tokens) < seq_len:
        tokens += [0] * (seq_len - len(tokens))
    return torch.tensor(tokens[:seq_len], dtype=torch.long)

def simple_detokenizer(tokens, text, max_vocab=1000):
    words = text.split()
    word_to_idx = {word: idx for idx, word in enumerate(set(words[:max_vocab]))}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    # convert each token id back to a word or <unk> if not found
    return [idx_to_word.get(t.item(), "<unk>") for t in tokens]

# load dataset
def prepare_dataset():
    dataset = load_dataset("tiny_shakespeare")["train"]
    text = dataset[0]["text"]
    return text

# split data into batches
def create_batches(text, batch_size=1):
    tokens = simple_tokenizer(text) # text to tokens
    seq_len = GPT2_124M_CONFIG["context_length"]
    num_tokens = len(tokens)
    num_batches = num_tokens // (batch_size * seq_len)
    # truncate tokens to fit into batches
    tokens = tokens[:num_batches * batch_size * seq_len]
    tokens = tokens.view(batch_size, num_batches * seq_len)
    # split into inputs except last token, targets shifted by 1
    inputs = tokens[:, :-1].view(batch_size, -1, seq_len)[:, :-1]
    targets = tokens[:, 1:].view(batch_size, -1, seq_len)[:, :-1]
    return inputs, targets

# training
def train_gpt(model, text, num_epochs=1, batch_size=1, learning_rate=1e-3):
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    inputs, targets = create_batches(text, batch_size)
    num_batches = input.size(1)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx in range(num_batches):
            input_batch = inputs[:, batch_idx, :]
            target_batch = targets[:, batch_size, :]
            optimizer.zero_grad()
            logits = model(input_batch) # run model to get predictions
            # loss between predictions and targets
            loss = criterion(logits.view(-1, GPT2_124M_CONFIG["vocab_size"]), target_batch.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # limit gradients to avoid explosions
            optimizer.step()
            total_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_batches}, Average Loss: {avg_loss:.4f}")

    return model

# text generation
def generate(model, input_text, text, max_len=20):
    model.eval()
    tokens = simple_tokenizer(input_text)
    input_ids = tokens.unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    output_tokens = input_ids[0]
    # tokens back to words and join into a string
    return " ".join(simple_detokenizer(output_tokens, text))

if __name__ == "__main__":
    model = GPT2Model(GPT2_124M_CONFIG)
    text = prepare_dataset()
    trained_model = train_gpt(model, text, num_epochs=1, batch_size=1)
    torch.save(trained_model.state_dict(), "gpt-2small_124m_cpu.pth")
    input_text = "To be or not to be"
    generated_text = generate(trained_model, input_text, text)
    print("GPT-2small:", generated_text)



