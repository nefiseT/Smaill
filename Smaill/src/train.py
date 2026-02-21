import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model import Smaill
from src.tokenizer import SimpleTokenizer

#load all data
with open('data/training_input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()
    
tokenizer = SimpleTokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype= torch.long)

#split data
n= int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

model = Smaill(tokenizer.vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

#training loop
batch_size = 64
for steps in range(20000):
    ix = torch.randint(0, len(train_data) - batch_size, (batch_size,))
    x = torch.stack([train_data[i:i+32] for i in ix])
    y = torch.stack([train_data[i+1:i+32+1] for i in ix])
    
    logits, loss = model(x,y)
    optimizer.zero_grad(set_to_none= True)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #gradient clipping
    optimizer.step()
    
    if steps % 500 == 0:
        context = torch.zeros((1,1), dtype=torch.long)
        sample = tokenizer.decode(model.generate(context, max_new_tokens=20)[0].tolist())      
        print(f"Step{steps}: loss{loss.item():.4f} | Sample: {sample}")
    
torch.save(model.state_dict(), "weights/smaill.pt")
print("model trained & weights saved...")


context = torch.zeros((1,1), dtype=torch.long)
print("\n---- texts ------")
print(tokenizer.decode(model.generate(context, max_new_tokens=100)[0].tolist()))
