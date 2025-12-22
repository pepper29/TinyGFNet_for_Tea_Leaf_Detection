
# add post processing
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from scripts.augmentation import get_test_transforms
import torchvision.transforms.functional as TF
from models.teaformer import TeaFormer
from models.teaformer2 import TeaFormer2
import argparse

# ─────────────────────────────────────────────────────────
# 1) Temperature Scaling wrapper
# ─────────────────────────────────────────────────────────
class ModelWithTemperature(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits / self.temperature

    def set_temperature(self, valid_loader: DataLoader, device: torch.device):
        self.to(device)
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        # Collect logits & labels on validation set
        logits_list, labels_list = [], []
        self.model.eval()
        with torch.no_grad():
            for imgs, labels in valid_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = self.model(imgs)
                logits_list.append(logits)
                labels_list.append(labels)
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        def _eval():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(_eval)
        return self

# ─────────────────────────────────────────────────────────
# 2) Class‐Wise Thresholding
# ─────────────────────────────────────────────────────────
def compute_class_thresholds(valid_loader: DataLoader, model: nn.Module, device: torch.device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)

    num_classes = all_probs.shape[1]
    thresholds = np.zeros(num_classes)
    for c in range(num_classes):
        precision, recall, thr = precision_recall_curve(
            (all_labels == c).astype(int), all_probs[:, c]
        )
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        thresholds[c] = thr[np.nanargmax(f1)]
    return thresholds

# ─────────────────────────────────────────────────────────
# 3) Bias/Offset Correction
# ─────────────────────────────────────────────────────────
def compute_bias_offsets(valid_loader: DataLoader, model: nn.Module, device: torch.device):
    model.eval()
    sums = None
    counts = None
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            probs = torch.softmax(model(imgs), dim=1).cpu().numpy()
            if sums is None:
                sums = np.zeros(probs.shape[1])
                counts = np.zeros(probs.shape[1])
            for i, l in enumerate(labels.cpu().numpy()):
                sums[l] += probs[i, l]
                counts[l] += 1
    avg_true_prob = sums / counts
    offsets = 1.0 - avg_true_prob
    return offsets

# ─────────────────────────────────────────────────────────
# 4) Main evaluation with TTA + Temperature + Threshold + Bias
# ─────────────────────────────────────────────────────────
def evaluate(model_path, val_dir, test_dir,
             batch_size=64, num_classes=7,
             mode="mobilenet", activation="gelu"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base model
    checkpoint = torch.load(model_path, map_location=device)
    t2_modes = {"lcaf", "mobilenet", "eformer", "mobilevit", "tiny-lcaf"}
    if mode in t2_modes:
        base_model = TeaFormer2(num_classes=num_classes, mode=mode, activation=activation).to(device)
    else:
        base_model = TeaFormer(num_classes=num_classes, mode=mode, activation=activation).to(device)
    base_model.load_state_dict(checkpoint, strict=False)
    base_model.eval()

    # Validation loader for calibration & thresholds
    valid_dataset = ImageFolder(val_dir, transform=get_test_transforms())
    valid_loader  = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # 1) Temperature scaling
    model_ts = ModelWithTemperature(base_model).set_temperature(valid_loader, device)

    # 2) Compute class thresholds & bias offsets
    thresholds = compute_class_thresholds(valid_loader, model_ts, device)
    offsets    = compute_bias_offsets(valid_loader,    model_ts, device)

    # Test loader
    test_dataset = ImageFolder(test_dir, transform=get_test_transforms())
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    y_true, y_pred = [], []

    # TTA transforms: orig, hflip, vflip, rot(+10), rot(-10)
    def tta_batch_probs(batch):
        outs = []
        for fn in [lambda x: x,
                   lambda x: TF.hflip(x),
                   lambda x: TF.vflip(x),
                   lambda x: TF.rotate(x, 10),
                   lambda x: TF.rotate(x, -10)]:
            aug = fn(batch)
            logits = model_ts(aug.to(device))
            outs.append(torch.softmax(logits, dim=1).cpu())
        return torch.stack(outs).mean(0).numpy()  # (B, C)

    with torch.no_grad():
        for images, labels in test_loader:
            y_true.extend(labels.numpy())
            probs = tta_batch_probs(images)

            # 3) Bias correction
            probs += offsets[np.newaxis, :]

            # 4) Class‐wise thresholding
            for sample_probs in probs:
                mask = sample_probs >= thresholds
                if mask.any():
                    pred = int(np.argmax(sample_probs * mask))
                else:
                    pred = int(np.argmax(sample_probs))
                y_pred.append(pred)

    # Classification report & confusion matrix
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=test_dataset.classes))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Save visualizations
    viz_dir = "/data1/seyong/metaverse/tealeaf2/outputs/visualizations"
    os.makedirs(viz_dir, exist_ok=True)

    # Confusion matrix plot
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(test_dataset.classes))
    plt.xticks(ticks, test_dataset.classes, rotation=45)
    plt.yticks(ticks, test_dataset.classes)
    thresh = cm.max()/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i,j], ha="center",
                     color="white" if cm[i,j]>thresh else "black")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{mode}_confusion_matrix_pp.png"))
    plt.close()

    # Precision/Recall/F1 plot
    report_dict = classification_report(y_true, y_pred,
                                        target_names=test_dataset.classes,
                                        output_dict=True)
    classes = test_dataset.classes
    precision = [report_dict[c]["precision"] for c in classes]
    recall    = [report_dict[c]["recall"]    for c in classes]
    f1_score  = [report_dict[c]["f1-score"]  for c in classes]
    x = np.arange(len(classes)); width = 0.25

    plt.figure(figsize=(10,6))
    plt.bar(x-width, precision, width, label='Precision')
    plt.bar(x,       recall,    width, label='Recall')
    plt.bar(x+width, f1_score,  width, label='F1-Score')
    plt.xlabel("Classes"); plt.ylabel("Score")
    plt.title("Metrics per Class")
    plt.xticks(x, classes, rotation=45)
    plt.ylim(0,1); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{mode}_classification_metrics_pp.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TeaFormer2 with post‐processing")
    parser.add_argument("--model_path", type=str,  required=True, help="Path to model checkpoint")
    parser.add_argument("--val_dir",    type=str,  required=True, help="Validation dataset directory")
    parser.add_argument("--test_dir",   type=str,  required=True, help="Test dataset directory")
    parser.add_argument("--mode",       type=str,  default="tiny-lcaf", choices=[
        "dynamic","concat","cnn","vit","bottleneck","bottleneck_freeze",
        "lcaf","mobilenet","eformer","mobilevit","tiny-lcaf"
    ], help="Model mode")
    parser.add_argument("--batch_size", type=int,  default=64, help="Batch size")
    parser.add_argument("--num_classes",type=int,  default=7,  help="Number of classes")
    parser.add_argument("--activation", type=str,  default="gelu",
                        choices=["softplus","gelu","leaky_relu","relu"], help="Activation")

    args = parser.parse_args()
    evaluate(
        model_path = args.model_path,
        val_dir    = args.val_dir,
        test_dir   = args.test_dir,
        batch_size = args.batch_size,
        num_classes= args.num_classes,
        mode       = args.mode,
        activation = args.activation
    )


'''
# 후처리 적용전
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from scripts.augmentation import get_test_transforms
import torchvision.transforms.functional as TF
from models.teaformer import TeaFormer
from models.teaformer2 import TeaFormer2
import argparse

def evaluate(model_path, test_dir,
             batch_size=64, num_classes=7,
             mode="mobilenet", activation="gelu"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    # TeaFormer2 모드 처리
    t2_modes = {"lcaf", "mobilenet", "eformer", "mobilevit", "tiny-lcaf","cnn", "vit"}
    if mode in t2_modes:
        model = TeaFormer2(
            num_classes=num_classes,
            mode=mode,
            activation=activation
        ).to(device)
    else:
        model = TeaFormer(
            num_classes=num_classes,
            mode=mode,
            activation=activation
        ).to(device)

    # state_dict 로드
    try:
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        if unexpected:
            print(f"Warning: Unexpected keys for mode '{mode}': {unexpected}")
        if missing:
            print(f"Warning: Missing keys for mode '{mode}': {missing}")
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
        print("모델 구조와 체크포인트가 일치하는지 확인하세요.")
        return

    model.eval()

    # 테스트 데이터 로드
    test_dataset = ImageFolder(test_dir, transform=get_test_transforms())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # TTA: 원본, 좌우반전, 상하반전
            images_hflip = TF.hflip(images.clone())
            images_vflip = TF.vflip(images.clone())
            tta_imgs = torch.cat([images, images_hflip, images_vflip], dim=0)  # (3B, C, H, W)

            logits = model(tta_imgs)
            probs = torch.softmax(logits, dim=1)  # (3B, num_classes)
            probs = probs.view(3, -1, probs.size(-1))  # (3, B, num_classes)
            avg_probs = probs.mean(dim=0)  # (B, num_classes)

            # 클래스 2 (index=1) 보정
            avg_probs[:, 1] += 0.03

            preds = torch.argmax(avg_probs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Classification Report
    report_text = classification_report(y_true, y_pred, target_names=test_dataset.classes)
    report_dict = classification_report(y_true, y_pred, target_names=test_dataset.classes, output_dict=True)

    print("\nClassification Report:")
    print(report_text)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # 시각화 저장
    viz_dir = "/data1/seyong/metaverse/tealeaf2/outputs/visualizations"
    os.makedirs(viz_dir, exist_ok=True)

    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(test_dataset.classes))
    plt.xticks(ticks, test_dataset.classes, rotation=45)
    plt.yticks(ticks, test_dataset.classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{mode}_confusion_matrix.png"))
    plt.close()

    # Precision/Recall/F1 Plot
    classes = test_dataset.classes
    precision = [report_dict[c]["precision"] for c in classes]
    recall = [report_dict[c]["recall"] for c in classes]
    f1_score = [report_dict[c]["f1-score"] for c in classes]
    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1_score, width, label='F1-Score')
    plt.xlabel("Classes")
    plt.ylabel("Score")
    plt.title("Classification Metrics per Class")
    plt.xticks(x, classes, rotation=45)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{mode}_classification_metrics.png"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TeaFormer Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_dir", type=str, required=True, help="Test dataset directory")
    parser.add_argument("--mode", type=str, default="cnn", choices=[
        "dynamic", "concat", "cnn", "vit",
        "bottleneck", "bottleneck_freeze",
        "lcaf", "mobilenet", "eformer", "mobilevit", "tiny-lcaf"
    ], help="Model mode to evaluate")
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--num_classes", type=int, default=7, help="Number of classes")
    parser.add_argument("--activation", type=str, default="gelu", choices=["softplus", "gelu", "leaky_relu", "relu"], help="Activation function")

    args = parser.parse_args()
    evaluate(
        model_path=args.model_path,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        mode=args.mode,
        activation=args.activation
    )

'''




'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from scripts.augmentation import get_test_transforms
import torchvision.transforms.functional as TF
from models.teaformer import TeaFormer
from models.teaformer2 import TeaFormer2
import argparse

def evaluate(model_path, test_dir,
             batch_size=64, num_classes=7,
             mode="mobilenet", activation="gelu"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    # TeaFormer2 모드 처리
    t2_modes = {"lcaf", "mobilenet", "eformer", "mobilevit", "tiny-lcaf"}
    if mode in t2_modes:
        model = TeaFormer2(
            num_classes=num_classes,
            mode=mode,
            activation=activation
        ).to(device)
    else:
        model = TeaFormer(
            num_classes=num_classes,
            mode=mode,
            activation=activation
        ).to(device)

    # state_dict 로드
    try:
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        if unexpected:
            print(f"Warning: Unexpected keys for mode '{mode}': {unexpected}")
        if missing:
            print(f"Warning: Missing keys for mode '{mode}': {missing}")
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
        print("모델 구조와 체크포인트가 일치하는지 확인하세요.")
        return

    model.eval()

    # 테스트 데이터 로드
    test_dataset = ImageFolder(test_dir, transform=get_test_transforms())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # TTA: 원본, 좌우반전, 상하반전
            images_hflip = TF.hflip(images.clone())
            images_vflip = TF.vflip(images.clone())
            tta_imgs = torch.cat([images, images_hflip, images_vflip], dim=0)  # (3B, C, H, W)

            logits = model(tta_imgs)
            probs = torch.softmax(logits, dim=1)  # (3B, num_classes)
            probs = probs.view(3, -1, probs.size(-1))  # (3, B, num_classes)
            avg_probs = probs.mean(dim=0)  # (B, num_classes)

            # 클래스 2 (index=1) 보정
            avg_probs[:, 1] += 0.03

            preds = torch.argmax(avg_probs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Classification Report
    report_text = classification_report(y_true, y_pred, target_names=test_dataset.classes)
    report_dict = classification_report(y_true, y_pred, target_names=test_dataset.classes, output_dict=True)

    print("\nClassification Report:")
    print(report_text)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # 시각화 저장
    viz_dir = "/data1/seyong/metaverse/tealeaf2/outputs/visualizations"
    os.makedirs(viz_dir, exist_ok=True)

    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(test_dataset.classes))
    plt.xticks(ticks, test_dataset.classes, rotation=45)
    plt.yticks(ticks, test_dataset.classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{mode}_confusion_matrix.png"))
    plt.close()

    # Precision/Recall/F1 Plot
    classes = test_dataset.classes
    precision = [report_dict[c]["precision"] for c in classes]
    recall = [report_dict[c]["recall"] for c in classes]
    f1_score = [report_dict[c]["f1-score"] for c in classes]
    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1_score, width, label='F1-Score')
    plt.xlabel("Classes")
    plt.ylabel("Score")
    plt.title("Classification Metrics per Class")
    plt.xticks(x, classes, rotation=45)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{mode}_classification_metrics.png"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TeaFormer Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_dir", type=str, required=True, help="Test dataset directory")
    parser.add_argument("--mode", type=str, default="cnn", choices=[
        "dynamic", "concat", "cnn", "vit",
        "bottleneck", "bottleneck_freeze",
        "lcaf", "mobilenet", "eformer", "mobilevit", "tiny-lcaf"
    ], help="Model mode to evaluate")
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--num_classes", type=int, default=7, help="Number of classes")
    parser.add_argument("--activation", type=str, default="gelu", choices=["softplus", "gelu", "leaky_relu", "relu"], help="Activation function")

    args = parser.parse_args()
    evaluate(
        model_path=args.model_path,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        mode=args.mode,
        activation=args.activation
    )
'''

'''
import sys
import os
# 프로젝트 루트 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from scripts.augmentation import get_test_transforms
import torchvision.transforms.functional as TF
from models.teaformer import TeaFormer
from models.teaformer2 import TeaFormer2

import argparse

def evaluate(model_path, test_dir,
             batch_size=64, num_classes=7,
             mode="mobilenet", activation="gelu"):
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    # TeaFormer2 계열 모드인지 확인
    t2_modes = {"lcaf", "mobilenet", "eformer", "mobilevit", "tiny-lcaf"}
    if mode in t2_modes:
        if mode == "tiny-lcaf":
            # tiny-lcaf는 m_proj/e_proj를 사용하므로 여기서 dim 추출
            m_proj_w = checkpoint["m_proj.weight"]  # [64, mobilenet_dim]
            e_proj_w = checkpoint["e_proj.weight"]  # [64, eformer_dim]
            model = TeaFormer2(
                num_classes=num_classes,
                mode="tiny-lcaf",
                activation=activation,
                fuse_dim=128,
                mobilenet_dim=m_proj_w.size(1),
                eformer_dim=e_proj_w.size(1)
            ).to(device)
        else:
            qw = checkpoint["fusion.q_proj.weight"]
            kw = checkpoint["fusion.k_proj.weight"]
            fuse_dim      = qw.size(0)
            mobilenet_dim = qw.size(1)
            eformer_dim   = kw.size(1)
    
            model = TeaFormer2(
                num_classes=num_classes,
                mode=mode,
                activation=activation,
                fuse_dim=fuse_dim,
                mobilenet_dim=mobilenet_dim,
                eformer_dim=eformer_dim
            ).to(device)

    # 3) state_dict 로드
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if unexpected:
        print(f"Warning: Unexpected keys for mode '{mode}': {unexpected}")
    if missing:
        print(f"Warning: Missing keys for mode '{mode}': {missing}")

    model.eval()

    # 테스트 데이터 로드
    test_dataset = ImageFolder(test_dir, transform=get_test_transforms())
    test_loader  = DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=False)

    y_true, y_pred = [], []
    
    # class 2 후처리 적용
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
    
            # TTA: 원본, 좌우반전, 상하반전
            images_hflip = TF.hflip(images.clone())
            images_vflip = TF.vflip(images.clone())
            tta_imgs = torch.cat([images, images_hflip, images_vflip], dim=0)  # (3B, C, H, W)
    
            logits = model(tta_imgs)
            probs = torch.softmax(logits, dim=1)  # (3B, num_classes)
            probs = probs.view(3, -1, probs.size(-1))  # (3, B, num_classes)
            avg_probs = probs.mean(dim=0)  # (B, num_classes)
    
            # 클래스 2 (index=1)의 logits 보정 (Brown Blight)
            avg_probs[:, 1] += 0.03  # ※ 클래스 인덱스는 0부터 시작하므로 class 2는 index 1
    
            preds = torch.argmax(avg_probs, dim=1)
    
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    
    """
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    """
    # Classification Report
    report_text = classification_report(
        y_true, y_pred, target_names=test_dataset.classes) # , digits=2
    report_dict = classification_report(
        y_true, y_pred,
        target_names=test_dataset.classes,
        output_dict=True) # , digits=4
    print("\nClassification Report:")
    print(report_text)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # 시각화 저장 폴더
    viz_dir = "/data1/seyong/metaverse/tealeaf2/outputs/visualizations"
    os.makedirs(viz_dir, exist_ok=True)

    # 1) Confusion Matrix 플롯
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(test_dataset.classes))
    plt.xticks(ticks, test_dataset.classes, rotation=45)
    plt.yticks(ticks, test_dataset.classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    cm_path = os.path.join(viz_dir, f"{mode}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to: {cm_path}")

    # 2) Precision/Recall/F1 플롯
    classes   = test_dataset.classes
    precision = [report_dict[c]["precision"] for c in classes]
    recall    = [report_dict[c]["recall"]    for c in classes]
    f1_score  = [report_dict[c]["f1-score"]  for c in classes]
    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x,      recall,    width, label='Recall')
    plt.bar(x + width, f1_score, width, label='F1-Score')
    plt.xlabel("Classes")
    plt.ylabel("Score")
    plt.title("Classification Metrics per Class")
    plt.xticks(x, classes, rotation=45)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    metrics_path = os.path.join(viz_dir,
                                f"{mode}_classification_metrics.png")
    plt.savefig(metrics_path)
    plt.close()
    print(f"Saved metrics plot to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate TeaFormer Model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--test_dir",   type=str, required=True,
                        help="Test dataset directory")
    parser.add_argument("--mode",       type=str, default="cnn",
        choices=[
            "dynamic", "concat", "cnn", "vit",
            "bottleneck", "bottleneck_freeze",
            "lcaf", "mobilenet", "eformer", "mobilevit", "tiny-lcaf"
        ],
        help="Model mode to evaluate")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Evaluation batch size")
    parser.add_argument("--num_classes",type=int, default=7,
                        help="Number of classes")
    parser.add_argument("--activation", type=str, default="leaky_relu",
        choices=["softplus", "gelu", "leaky_relu", "relu"],
        help="Activation function used")

    args = parser.parse_args()

    evaluate(
        args.model_path, args.test_dir,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        mode=args.mode,
        activation=args.activation
    )

'''