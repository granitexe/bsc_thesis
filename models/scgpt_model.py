from models.base_model import BaseModel
import torch.nn as nn

class SCGPTModel(BaseModel):
    def __init__(self, num_layers=6, hidden_size=256):
        super(SCGPTModel, self).__init__()
        self.layers = nn.ModuleList([nn.Transformer(hidden_size, nhead=8) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
