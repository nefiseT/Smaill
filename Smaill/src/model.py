import torch 
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """Single attention head - optimized"""
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
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        out = self.proj(out)
        return out


class Smaill(nn.Module): 
    def __init__(self, vocab_size, block_size=64, n_embd=256, n_heads=4):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        
        # Larger embedding table for better capacity
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Single attention head (faster but still effective)
        self.attention = Head(n_embd // n_heads, n_embd, block_size)
        
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, n_embd * 2),
            nn.GELU(),
            nn.Linear(n_embd * 2, n_embd),
            nn.Dropout(0.1),
        )
        
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        if T > self.block_size:
            idx = idx[:, -self.block_size:]
            T = self.block_size
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) 
        x = tok_emb + pos_emb
        
        # attention and feed-forward (residual connections) - after this output were midly meaningful
        x = x + self.attention(x)
        x = x + self.ffwd(x)
        
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):    
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
        
        return idx


if __name__ == "__main__":
    print("✓ Smaill model.py loaded successfully!")

    model = Smaill(vocab_size=100, block_size=32, n_embd=64)
    x = torch.randint(0, 100, (2, 10))  # Batch of 2, 10 tokens
    logits, loss = model(x)
    print(f"✓ Forward pass OK: input {x.shape} -> output {logits.shape}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters())}")
