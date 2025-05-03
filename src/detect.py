import os
from PIL import Image
import torch
from torchvision import transforms
from src.model import EfficientNetClassifier

def predict_folder(folder_path, model_path="../models/best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(root, file)
                image = Image.open(img_path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    prob = output.item()
                    label = int(prob > 0.5)

                print(f"{img_path}: Label={label}, Probability={prob:.4f}")

if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    predict_folder(folder)