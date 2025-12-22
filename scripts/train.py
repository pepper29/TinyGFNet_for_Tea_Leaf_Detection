# train.py
# 로그 저장, 진행상황 시각화
# 체크포인트 저장방식 변경(10epoch-> best저장, 실험종료-> final 저장)
# loss 5회이상 감소 없을 시 lr 감소 추가
# 재시작 기능추가
# 파라미터 수 나타냄
import sys
import os
# 프로젝트 루트 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.teaformer2 import TeaFormer2
from scripts.dataset import TeaLeafDataset
from scripts.augmentation import get_train_transforms, get_test_transforms
from config.config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpoint 저장 경로
ckpt_dir = "/data1/seyong/metaverse/tealeaf2/outputs/checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

# 로그 저장 경로
log_dir = "/data1/seyong/metaverse/tealeaf2/outputs/logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"{config['model_mode']}_{config['num_epochs']}_exp.txt")
log_file = open(log_path, "w")

# MixUp 함수
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

# DataLoader
base = "/data1/seyong/metaverse/tealeaf2/data/split"
train_loader = DataLoader(
    TeaLeafDataset(f"{base}/train", transform=get_train_transforms()),
    batch_size=config["batch_size"], shuffle=True, num_workers=4)
val_loader = DataLoader(
    TeaLeafDataset(f"{base}/val", transform=get_test_transforms()),
    batch_size=config["batch_size"], shuffle=False, num_workers=4)

# Model
model = TeaFormer2(
    num_classes=config["num_classes"],
    mode=config["model_mode"],
    activation=config["activation"]
).to(device)
print(f"[Debug] TeaFormer2 Mode: {model.mode}")
# 파라미터 수·FLOPs 계산
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params}")
from ptflops import get_model_complexity_info
macs, _ = get_model_complexity_info(
    model,
    (3, config["image_size"], config["image_size"]),
    as_strings=True, print_per_layer_stat=False, verbose=False
)
print(f"Model complexity: {macs} MACs")

# Criterion·Optimizer·Scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, verbose=True)

best_val_acc = 0.0

for epoch in range(config["num_epochs"]):
    model.train()
    total_loss = 0.0
    total_correct = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        imgs, labels = imgs.to(device), labels.to(device)

        mixed_imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=0.4)
        optimizer.zero_grad()
        outputs = model(mixed_imgs)
        loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (lam * (preds == y_a).float() + (1 - lam) * (preds == y_b).float()).sum().item()

    train_loss = total_loss / len(train_loader.dataset)
    train_acc = total_correct / len(train_loader.dataset)

    model.eval()
    val_correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            val_correct += (model(imgs).argmax(dim=1) == labels).sum().item()
    val_acc = val_correct / len(val_loader.dataset)

    scheduler.step(val_acc)

    log_line = (
        f"Epoch {epoch+1}/{config['num_epochs']}, "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
        f"Val Acc: {val_acc:.4f}\n"
    )
    print(log_line.strip())
    log_file.write(log_line)
    log_file.flush()

    # best 모델을 outputs/checkpoints에 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path = os.path.join(ckpt_dir, f"best_{config['model_mode']}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"New best model saved to: {save_path}")

# 로그 파일 닫기
log_file.close()