import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model import Smaill
from src.tokenizer import SimpleTokenizer

# Load all data
with open('data/training_input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
tokenizer = SimpleTokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Total tokens: {len(data)}")

# Split data
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Optimized model - larger embeddings, single attention head for speed
model = Smaill(
    vocab_size=tokenizer.vocab_size,
    block_size=64,
    n_embd=256,
    n_heads=4
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Use fused optimizer for speed
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

# Training hyperparameters - larger batch for speed
batch_size = 128
block_size = 64

# Training loop
for steps in range(10000):
    # Sample random starting positions
    ix = torch.randint(0, len(train_data) - block_size, (batch_size,))
    
    # Create batches
    x = torch.stack([train_data[i:i+block_size] for i in ix])
    y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
    
    logits, loss = model(x, y)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if steps % 200 == 0:
        context = torch.zeros((1, 1), dtype=torch.long)
        sample = tokenizer.decode(model.generate(context, max_new_tokens=50)[0].tolist())      
        print(f"Step {steps}: loss {loss.item():.4f} | Sample: {sample}")

torch.save(model.state_dict(), "weights/smaill.pt")
print("Model trained & weights saved...")

# Generate final sample
context = torch.zeros((1, 1), dtype=torch.long)
print("\n---- Generated Text ------")
print(tokenizer.decode(model.generate(context, max_new_tokens=200)[0].tolist()))
torch.save(model.state_dict(), "weights/smaill.pt")
print("Model trained & weights saved...")

# Generate final sample
context = torch.zeros((1, 1), dtype=torch.long)
print("\n---- Generated Text ------")
print(tokenizer.decode(model.generate(context, max_new_tokens=200)[0].tolist()))
