# Lightweight Dual-Backbone Framework for Tea Leaf Disease Detection

> **📢 News** > This paper has been accepted by **IEEE International Symposium on Consumer Technology (ISCT) 2025**.  
> **Paper Link:** [TBD]

---

## 📌 Introduction
Tea diseases can significantly impact crop yield and quality. To address this, we propose **TinyGFNet**, a lightweight dual-backbone framework designed for efficient and accurate tea leaf disease detection in mobile and edge computing environments.

This project implements the official code for the paper:  
**"Lightweight Dual-Backbone Framework for Tea Leaf Disease Detection"**.

<p align="center">
  <img src="assets/teaformer_model.png" alt="TinyGFNet Architecture" width="800"/>
</p>

## ✨ Key Features
* **Dual-Backbone Architecture:** Combines **MobileNetV3** (Local Features) and **EfficientFormer** (Global Features) to maximize feature extraction capability.
* **Gated Fusion Module:** Effectively fuses heterogeneous features from CNN and Transformer branches using a gate mechanism.
* **Lightweight & Efficient:** Achieves state-of-the-art performance with significantly lower parameters and FLOPs compared to ViT.
  * **Parameters:** 12.88M
  * **FLOPs:** 1.34G
  * **F1-Score:** 0.95

## 📂 Project Structure
```bash
.
├── assets/                # Images for README (Model architecture, Results)
├── config/                # Configuration files
├── models/                # TinyGFNet model implementation
│   └── teaformer.py       # Main model file
├── scripts/               # Training & Inference scripts
│   ├── train.py
│   ├── inference.py
│   ├── dataset.py
│   └── ...
├── requirements.txt       # Python dependencies
└── README.md
🚀 Getting Started1. InstallationClone the repository and install dependencies.Bashgit clone [https://github.com/pepper29/TinyGFNet.git](https://github.com/pepper29/TinyGFNet.git)
cd TinyGFNet
pip install -r requirements.txt
2. Dataset PreparationStructure your dataset as follows:data/
  ├── train/
  │   ├── class_1/
  │   ├── class_2/
  │   └── ...
  └── val/
      ├── class_1/
      └── ...
3. TrainingTo train the TinyGFNet model on your dataset:Bashpython scripts/train.py --config config/config.py
4. InferenceTo detect diseases from a single image:Bashpython scripts/inference.py --image_path sample_leaf.jpg --checkpoint outputs/best_model.pth
📊 Experimental ResultsModelParams (M)FLOPs (G)AccuracyCNN11.241.820.92ViT86.0012.020.95MobileNetV30.700.020.80TinyGFNet (Ours)12.881.340.95<p align="center"><img src="assets/best_confusion_matrix.png" alt="Confusion Matrix" width="45%"/><img src="assets/demo_result.gif" alt="Demo Result" width="45%"/></p>📜 CitationIf you find this work useful, please cite our paper:코드 스니펫@inproceedings{jin2025tinygfnet,
  title={Lightweight Dual-Backbone Framework for Tea Leaf Disease Detection},
  author={Seyong Jin and [Co-authors]},
  booktitle={2025 IEEE International Symposium on Consumer Technology (ISCT)},
  year={2025},
  note={Accepted}
}
📝 LicenseThis project is licensed under the MIT License.
