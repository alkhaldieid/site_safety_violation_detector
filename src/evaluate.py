import torch
from sklearn.metrics import classification_report
from src.data_loader import get_dataloaders
from src.model import EfficientNetClassifier

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, _ = get_dataloaders("/data/home/alkhaldieid/repos/site_safety_violation_detector/data/Worksite-Safety-Monitoring-Dataset")

    model = EfficientNetClassifier().to(device)
    model.load_state_dict(torch.load("/data/home/alkhaldieid/repos/site_safety_violation_detector/models/best_model.pth", map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print(classification_report(y_true, y_pred))
    