import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Each model must implement the forward method")

    def train_model(self, train_loader, criterion, optimizer, epochs=10, checkpoint_path=None):
        self.train()
        best_loss = float('inf')
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Loss: {avg_loss}")

            # Save checkpoint
            if checkpoint_path:
                torch.save(self.state_dict(), checkpoint_path)

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.state_dict(), checkpoint_path.replace('checkpoint', 'best_model'))

    def evaluate_model(self, test_loader, criterion):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy}%")
        return accuracy
