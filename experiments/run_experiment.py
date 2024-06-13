import yaml
from train.train_model import get_model
from data.datasets import load_dataset
import torch.optim as optim
import torch.nn as nn

with open('experiments/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

train_loader, test_loader = load_dataset()
model = get_model(config['model'])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if config.get('train', True):
    model.train_model(train_loader, criterion, optimizer, epochs=config['epochs'], checkpoint_path=config['checkpoint_path'])

accuracy = model.evaluate_model(test_loader, criterion)
