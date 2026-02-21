# torch-stoi 0.2.3   -> string to int

class SimpleTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch:i for i,ch in enumerate(self.chars)}
        self.itos = {i:ch for i,ch in enumerate(self.chars)}
        
    def encode(self,s):
        return [self.stoi[c] for c in s if c in self.stoi]
    
    def decode(self, l):
        return ''.join([self.itos[i] for i in l])
