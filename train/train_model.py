import torch
import torch.optim as optim
import torch.nn as nn
from data.datasets import load_dataset
from models.scgpt_model import SCGPTModel
from models.hyenadna_model import HyenaDNAModel
from models.mamba_model import MambaModel
import argparse

def get_model(model_name):
    if model_name == 'scGPT':
        return SCGPTModel()
    elif model_name == 'HyenaDNA':
        return HyenaDNAModel()
    elif model_name == 'MAMBA':
        return MambaModel()
    else:
        raise ValueError(f"Unknown model name {model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model name: scGPT, HyenaDNA, MAMBA')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--checkpoint', type=str, default='train/checkpoints/checkpoint.pt', help='Checkpoint path')
    args = parser.parse_args()

    train_loader, test_loader = load_dataset()
    model = get_model(args.model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train_model(train_loader, criterion, optimizer, epochs=args.epochs, checkpoint_path=args.checkpoint)
    model.evaluate_model(test_loader, criterion)
