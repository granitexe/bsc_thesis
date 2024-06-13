import torch
from torch.utils.data import DataLoader, TensorDataset

def load_dataset():
    # Placeholder for actual dataset loading and processing
    train_data = torch.randn(100, 10, 256)
    train_labels = torch.randint(0, 2, (100,))
    test_data = torch.randn(20, 10, 256)
    test_labels = torch.randint(0, 2, (20,))

    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    return train_loader, test_loader
