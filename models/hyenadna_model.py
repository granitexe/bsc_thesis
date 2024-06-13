from models.base_model import BaseModel
import torch.nn as nn

class HyenaDNAModel(BaseModel):
    def __init__(self, num_layers=2, hidden_size=128):
        super(HyenaDNAModel, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x
