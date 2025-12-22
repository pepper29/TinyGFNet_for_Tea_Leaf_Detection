# File: scripts/augmentation.py

import os
import random
import shutil
from PIL import Image
from torchvision import transforms
# 상대임포트: scripts 폴더 내 __init__.py가 있어야 함.
from preprocessing import apply_preprocessing, adaptive_histogram_equalization, z_score_normalization

def get_train_transforms():
    """
    증강에서는 잎의 색상 정보를 보존하기 위해 ColorJitter 없이
    기하학적 변형만 적용합니다.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        # ↓ 사전학습된 백본이 기대하는 ImageNet 정규화
        transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                             std=[0.229, 0.224, 0.225])
    ])

def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                             std=[0.229, 0.224, 0.225])
    ])

def augment_and_save(raw_data_dir, processed_data_dir, target_count_per_class=1000):
    """
    각 클래스별 (raw_data_dir/<class>/) 원본 이미지에 대해
    먼저 전처리(apply_preprocessing)를 수행하여 processed_data_dir/<class>/에 저장하고,
    파일명이 {클래스명}_{번호:05}.jpg 형식으로 통일되도록 합니다.
    
    이후, 각 클래스의 이미지 수가 target_count_per_class에 미치지 않으면,
    전처리된 이미지 중 랜덤하게 선택하여 augmentation을 적용한 결과도 동일한 파일명 형식으로 추가 저장합니다.
    """
    os.makedirs(processed_data_dir, exist_ok=True)
    train_transforms = get_train_transforms()
    to_pil = transforms.ToPILImage()
    
    for cls in os.listdir(raw_data_dir):
        class_raw_dir = os.path.join(raw_data_dir, cls)
        if not os.path.isdir(class_raw_dir):
            continue

        class_proc_dir = os.path.join(processed_data_dir, cls)
        os.makedirs(class_proc_dir, exist_ok=True)
        
        # 순번 카운터 (1부터 시작)
        counter = 1
        
        # 이미지 파일 목록: JPG/JPEG/PNG
        raw_images = [f for f in os.listdir(class_raw_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # 1. 전처리 적용 후 원본 이미지 저장
        for img_name in raw_images:
            src_path = os.path.join(class_raw_dir, img_name)
            try:
                image = Image.open(src_path).convert("RGB")
            except Exception as e:
                print(f"Error opening {src_path}: {e}")
                continue
            preprocessed_image = apply_preprocessing(image)
            new_filename = f"{cls}_{counter:05}.jpg"
            dst_path = os.path.join(class_proc_dir, new_filename)
            preprocessed_image.save(dst_path)
            counter += 1
        
        # 2. target_count_per_class까지 augmentation
        # processed_images: 이미 저장된 전처리된 이미지 파일명 목록
        processed_images = [f for f in os.listdir(class_proc_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        while counter <= target_count_per_class:
            # processed_images 중 랜덤 선택하여 augmentation 적용
            base_file = random.choice(processed_images)
            base_path = os.path.join(class_proc_dir, base_file)
            try:
                image = Image.open(base_path).convert("RGB")
            except Exception as e:
                print(f"Error opening {base_path}: {e}")
                continue
            aug_tensor = train_transforms(image)
            aug_image = to_pil(aug_tensor)
            new_filename = f"{cls}_{counter:05}.jpg"
            dst_path = os.path.join(class_proc_dir, new_filename)
            try:
                aug_image.save(dst_path)
                counter += 1
                processed_images.append(new_filename)
            except Exception as e:
                print(f"Error saving {dst_path}: {e}")
        
        print(f"Class '{cls}': {counter-1} images saved in {class_proc_dir}")

def split_dataset(processed_data_dir, split_data_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    processed_data_dir 아래 각 클래스 폴더의 전처리+증강 이미지를
    split_data_dir 아래에 train, val, test 폴더로 7:1:2 비율로 분할하여 저장합니다.
    파일명은 그대로 유지됩니다.
    """
    os.makedirs(split_data_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(split_data_dir, split)
        os.makedirs(split_path, exist_ok=True)
    
    for cls in os.listdir(processed_data_dir):
        class_dir = os.path.join(processed_data_dir, cls)
        if not os.path.isdir(class_dir):
            continue
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        images.sort()  # 번호 순 정렬
        random.shuffle(images)  # 필요 시 셔플
        n = len(images)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        train_files = images[:train_end]
        val_files = images[train_end:val_end]
        test_files = images[val_end:]
        
        for split, file_list in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
            dest_dir = os.path.join(split_data_dir, split, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for file in file_list:
                src = os.path.join(class_dir, file)
                dst = os.path.join(dest_dir, file)
                shutil.copy(src, dst)
        print(f"Class '{cls}': split into {len(train_files)} train, {len(val_files)} val, {len(test_files)} test images.")

if __name__ == '__main__':
    # 원본 데이터는 /data1/seyong/metaverse/tealeaf/data/teaLeafBD/teaLeafBD에, 
    # 클래스별로 정리되어 있다고 가정합니다.
    raw_data = "/data1/seyong/metaverse/tealeaf/data/teaLeafBD/teaLeafBD"
    # 전처리 및 증강된 데이터를 저장할 폴더
    processed_data = "/data1/seyong/metaverse/tealeaf2/data/preprossed_data"
    augment_and_save(raw_data, processed_data, target_count_per_class=1000)
    
    # 전처리+증강 데이터를 7:1:2 비율로 분할하여 저장
    split_data = "/data1/seyong/metaverse/tealeaf2/data/split"
    split_dataset(processed_data, split_data, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
