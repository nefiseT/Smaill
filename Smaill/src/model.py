import torch 
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """Single attention"""
    def __init__(self, head_size, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.proj = nn.Linear(head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # computes attention scores
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        out = self.proj(out)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out_2 = torch.cat([h(x) for h in self.heads], dim=-1)
        out_2 = self.dropout(self.proj(out))
        return out_2

class Block(nn.Module):
    """ Attention + feed forward = more logic """
    def __init__(self, n_embd, n_heads, block_size, dropout=0.2):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, n_embd, block_size, dropout)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Residual connections (x + ...) help gradients flow during training
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Smaill(nn.Module): 
    def __init__(self, vocab_size, block_size=128, n_embd=256, n_heads=8, n_layers=4, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # The Stack of Layers
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_heads, block_size, dropout) for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # Check to avoid position embedding overflow
        if T > self.block_size:
            idx = idx[:, -self.block_size:]
            T = self.block_size
            
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) 
        x = tok_emb + pos_emb
        
        x = self.blocks(x) # Pass through all 4 layers
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=20):
        self.eval()     #switch to eval
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()    #switch to train
        return idx


if __name__ == "__main__":
    print("✓ Smaill model.py loaded successfully!")

    model = Smaill(vocab_size=100, block_size=32, n_embd=64)
    x = torch.randint(0, 100, (2, 10))  # Batch of 2, 10 tokens
    logits, loss = model(x)
    print(f"✓ Forward pass OK: input {x.shape} -> output {logits.shape}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters())}")
