from models.base_model import BaseModel
import torch.nn as nn

class MambaModel(BaseModel):
    def __init__(self, num_layers=4, hidden_size=256):
        super(MambaModel, self).__init__()
        self.layers = nn.ModuleList([nn.RNN(hidden_size, hidden_size, num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
        return x
