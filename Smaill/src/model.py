import torch 
import torch.nn as nn
from torch.nn import functional as F

class Smaill(nn.Module): 
    def __init__(self, vocab_size):
        super().__init__()
        self.n_embd = 64    #small vectorr size
        self.block_size = 32       #short memory, token length

        self.token_embedding_table = nn.Embedding(vocab_size, self.n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd)
        self.lm_head = nn.Linear(self.n_embd, vocab_size)


    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)   #B,T,C,
        pos_emb = self.position_embedding_table(torch.arange(T)) 
        x = tok_emb + pos_emb
        logits = self.lm_head(x)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx