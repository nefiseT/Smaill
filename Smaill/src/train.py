import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model import Smaill
from src.tokenizer import SimpleTokenizer

# Check CUDA availability more thoroughly
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

# Check if GPU is accessible
if torch.cuda.is_available():
    try:
        # Test if we can actually use the GPU
        _ = torch.zeros(1).cuda()
        device = torch.device("cuda")
        print(f"GPU is accessible: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"GPU found but not accessible: {e}")
        device = torch.device("cpu")
else:
    # Check if there are GPUs that might be hidden
    print("CUDA not available. Checking potential issues...")
    print("- Is NVIDIA GPU installed? Check: nvidia-smi")
    print("- Is PyTorch with CUDA installed? Try: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    print("- Is CUDA_VISIBLE_DEVICES set? Current value:", os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'))
    device = torch.device("cpu")

print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Enable cuDNN benchmark for optimized performance
    torch.backends.cudnn.benchmark = True
    
    # Try to use torch.compile for PyTorch 2.0+ (significant speedup)
    try:
        torch.compile
        use_compile = True
        print("torch.compile available - will be used for optimization")
    except AttributeError:
        use_compile = False
        print("torch.compile not available - using standard execution")
else:
    use_compile = False

with open('data/training_input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
tokenizer = SimpleTokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Total tokens: {len(data)}")

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Optimized model - larger embeddings, single attention head for speed
model = Smaill(
    vocab_size=tokenizer.vocab_size,
    block_size=64,
    n_embd=256,
    n_heads=8
).to(device)  # Move model to GPU!

# Apply torch.compile for PyTorch 2.0+ speedup
if use_compile:
    model = torch.compile(model)
    print("Model compiled for GPU acceleration")

print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

# Training hyperparameters - larger batch for speed
batch_size = 128
block_size = 64

for steps in range(10000):
    # Sample random starting positions
    ix = torch.randint(0, len(train_data) - block_size, (batch_size,))
    
    # Create batches and MOVE TO GPU! (non_blocking for async transfer)
    x = torch.stack([train_data[i:i+block_size] for i in ix]).to(device, non_blocking=True)
    y = torch.stack([train_data[i+1:i+block_size+1] for i in ix]).to(device, non_blocking=True)
    
    logits, loss = model(x, y)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if steps % 200 == 0:
        # Move context to same device for sampling
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        sample = tokenizer.decode(model.generate(context, max_new_tokens=50)[0].cpu().tolist())      
        print(f"Step {steps}: loss {loss.item():.4f} | Sample: {sample}")

torch.save(model.state_dict(), "weights/smaill.pt")
print("Model trained & weights saved...")

# Generate final sample
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\n---- Generated Text ------")
print(tokenizer.decode(model.generate(context, max_new_tokens=200)[0].cpu().tolist()))
