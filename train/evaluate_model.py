import torch
from utils.save_load import load_model
from data.datasets import load_dataset
import argparse

def evaluate_model(model_name, checkpoint_path):
    train_loader, test_loader = load_dataset()

    model = load_model(model_name, checkpoint_path)

    criterion = torch.nn.CrossEntropyLoss()
    model.evaluate_model(test_loader, criterion)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model name: scGPT, HyenaDNA, MAMBA')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    args = parser.parse_args()

    evaluate_model(args.model, args.checkpoint)
