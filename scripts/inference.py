import os
import torch
from PIL import Image
from scripts.preprocessing import apply_preprocessing
from scripts.augmentation import get_test_transforms
from models.teaformer import TeaFormer

# 클래스 이름은 dataset 폴더 이름 순서를 따름 (예시)
CLASS_NAMES = [
    "Tea algal leaf spot", "Brown Blight", "Gray Blight", "Helopolis",
    "Red spider", "Green mirid bug", "Healthy Leaf"
]

def predict(image_path, model_path="outputs/checkpoints/best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TeaFormer(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    image = Image.open(image_path).convert("RGB")
    image = apply_preprocessing(image)
    transform = get_test_transforms()
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        pred_idx = outputs.argmax(dim=1).item()
        pred_class = CLASS_NAMES[pred_idx]
        print(f"Predicted Class: {pred_class}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python inference.py <image_path>")
    else:
        predict(sys.argv[1])