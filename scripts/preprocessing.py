# File: scripts/preprocessing.py

import cv2
import numpy as np
from PIL import Image

def min_max_normalization(img):
    """
    보조적인 기능: 이미지의 최소/최대 값을 이용해 0~1 범위로 정규화합니다.
    보통 z‑score 정규화와 함께 사용하지 않는 것을 권장합니다.
    """
    img = np.array(img).astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def z_score_normalization(img):
    """
    z‑score normalization: 이미지의 평균을 0, 표준편차를 1로 만드는 정규화 방법입니다.
    잎의 질병 분류에서는 대비 및 텍스처 차이를 강조하는 데 유용할 수 있습니다.
    """
    img = np.array(img).astype(np.float32)
    mean, std = img.mean(), img.std()
    return (img - mean) / (std + 1e-8)

def adaptive_histogram_equalization(img):
    """
    CLAHE (adaptive histogram equalization)을 사용하여
    이미지의 대비를 향상시킵니다.
    이는 잎의 병변과 변색된 영역을 강조하는 데 도움이 됩니다.
    """
    img = np.array(img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return img_eq

def median_filter(img, ksize=3):
    """
    Median filtering으로 노이즈를 줄입니다.
    """
    img = np.array(img)
    return cv2.medianBlur(img, ksize)

def sharpen(img):
    """
    샤프닝 필터를 적용하여 이미지의 경계를 강조합니다.
    """
    img = np.array(img)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def apply_preprocessing(pil_image):
    """
    전처리 함수: 아래 단계들을 순차적으로 적용합니다.
    
    1. Adaptive Histogram Equalization: 대비를 향상시킵니다.
    2. 샤프닝과 Median Filtering: 노이즈를 줄이고 경계를 보강합니다.
    3. z‑score normalization: 전체적인 정규화를 통해 학습의 안정성을 높입니다.
    4. (Optional) Min‑Max normalization: 일반적으로 z‑score 정규화 후 사용하지 않으므로 주석 처리합니다.
    5. 최종적으로 0~255 범위의 uint8 이미지로 변환하여 PIL 이미지로 반환합니다.
    """
    img = np.array(pil_image)
    
    # 1. 대비 향상을 위한 Adaptive Histogram Equalization
    img = adaptive_histogram_equalization(img)
    
    # 2. 이미지 디테일 보강: 샤프닝과 Median Filtering
    img = sharpen(img)
    img = median_filter(img, ksize=3)
    
    # 3. z‑score normalization 적용
    img = z_score_normalization(img)
    
    # 4. Optional: Min‑Max normalization (병행 사용하지 않음)
    # img = min_max_normalization(img)
    
    # 최종적으로 0~255 범위의 uint8 타입으로 변환
    # (z‑score normalization 결과는 음수 및 1보다 큰 값이 존재할 수 있으므로, 여기서 다시 스케일 조정)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0
    img = img.astype(np.uint8)
    return Image.fromarray(img)
