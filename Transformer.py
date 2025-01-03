#Needed to install pyTorch
#Commands if there is a pytorch: pip list | findstr torch
#For installing pytorch: pip install torch torchvision torchaudio
import torch
import torch.nn as nn
import math

class InputEmbedding(torch.nn.Module):
    def __init__(self, d_model , vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
        