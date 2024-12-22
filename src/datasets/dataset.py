import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from albumentations import (
    Compose, Resize, HorizontalFlip, VerticalFlip,
    RandomBrightnessContrast, RandomRotate90,
    GaussianBlur, GaussNoise, ColorJitter
)

class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None, train=True, val=False, img_size=128, use_augmentation=False, subset_fraction=1.0):
        """
        Args:
            csv_file (str): 데이터셋 CSV 파일 경로
            transform (callable, optional): 데이터 증강을 위한 변환
            train (bool): 학습/테스트 모드
            val (bool): 검증 데이터셋 여부
            img_size (int): 이미지 크기
            use_augmentation (bool): 데이터 증강 사용 여부
            subset_fraction (float): 사용할 데이터의 비율 (0.0 ~ 1.0)
        """
        # CSV 파일 읽기
        self.data = pd.read_csv(csv_file)
        
        # 전체 데이터 중 일부만 사용
        if subset_fraction < 1.0:
            self.data = self.data.sample(frac=subset_fraction, random_state=42)
        
        if val:
            # 20%를 검증 데이터로 사용
            self.data = self.data.sample(frac=0.2, random_state=42)
        elif train:
            # 80%를 학습 데이터로 사용
            val_data = self.data.sample(frac=0.2, random_state=42)
            self.data = self.data.drop(val_data.index)
            
        self.transform = transform
        self.train = train
        self.img_size = img_size
        
        if transform is None:
            if train and not val and use_augmentation:
                self.transform = Compose([
                    Resize(img_size, img_size, always_apply=True),
                    HorizontalFlip(p=0.5),           # 좌우 반전
                    VerticalFlip(p=0.5),             # 상하 반전
                    RandomRotate90(p=0.5),           # 90도 회전
                    RandomBrightnessContrast(
                        brightness_limit=0.2,         # 밝기 변화 범위
                        contrast_limit=0.2,          # 대비 변화 범위
                        p=0.5
                    ),
                    GaussianBlur(
                        blur_limit=(3, 7),           # 블러 커널 크기
                        p=0.3
                    ),
                    GaussNoise(
                        var_limit=(10.0, 50.0),      # 노이즈 강도
                        p=0.3
                    ),
                    ColorJitter(
                        brightness=0.2,              # 밝기 변화
                        contrast=0.2,                # 대비 변화
                        saturation=0.2,              # 채도 변화
                        hue=0.1,                     # 색조 변화
                        p=0.3
                    )
                ])
            else:
                self.transform = Compose([
                    Resize(img_size, img_size, always_apply=True)
                ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.train:
            input_path = self.data.iloc[idx]['input_image_path']
            gt_path = self.data.iloc[idx]['gt_image_path']
            
            # 상대 경로를 절대 경로로 변환
            input_path = os.path.join('data', input_path.lstrip('./'))
            gt_path = os.path.join('data', gt_path.lstrip('./'))
            
            # 이미지 로드 최적화
            input_img = cv2.imread(input_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            
            gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            
            # 데이터 증강 적용
            if self.transform:
                augmented = self.transform(image=input_img, mask=gt_img)
                input_img = augmented['image']
                gt_img = augmented['mask']
            
            # 정규화 (0-1 범위로)
            input_img = input_img.astype(np.float32) / 255.0
            gt_img = gt_img.astype(np.float32) / 255.0
            
            # (H, W, C) -> (C, H, W)
            input_img = np.ascontiguousarray(input_img.transpose(2, 0, 1))
            gt_img = np.ascontiguousarray(gt_img.transpose(2, 0, 1))
            
            return input_img, gt_img
        else:
            input_path = self.data.iloc[idx]['input_image_path']
            input_path = os.path.join('data', input_path.lstrip('./'))
            
            input_img = cv2.imread(input_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                augmented = self.transform(image=input_img)
                input_img = augmented['image']
            
            input_img = input_img.astype(np.float32) / 255.0
            input_img = np.ascontiguousarray(input_img.transpose(2, 0, 1))
            
            return input_img