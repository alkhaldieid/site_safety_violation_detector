import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import get_dataloaders
from src.model import EfficientNetClassifier
from src.utils import plot_history, save_model

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=25):
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    for epoch in range(epochs):
        model.train()
        train_loss, train_acc, total = 0.0, 0.0, 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = (outputs > 0.5).float()
            train_loss += loss.item() * inputs.size(0)
            train_acc += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss /= total
        train_acc /= total

        model.eval()
        val_loss, val_acc, val_total = 0.0, 0.0, 0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.float().unsqueeze(1).to(device)
                val_outputs = model(val_inputs)
                loss = criterion(val_outputs, val_labels)
                preds = (val_outputs > 0.5).float()
                val_loss += loss.item() * val_inputs.size(0)
                val_acc += (preds == val_labels).sum().item()
                val_total += val_labels.size(0)

        val_loss /= val_total
        val_acc /= val_total

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    return history

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _, classes = get_dataloaders("../data/Worksite-Safety-Monitoring-Dataset")
    model = EfficientNetClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    history = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=25)
    plot_history(history)
    save_model(model, "../models/best_model.pth")