import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as np

from datasets.dataset import ImageDataset
from models.unet import UNet

def save_predictions(model, test_loader, device, output_dir):
    """테스트 데이터에 대한 예측을 수행하고 결과를 저장합니다."""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, inputs in enumerate(tqdm(test_loader, desc='Generating predictions')):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # 배치의 각 이미지에 대해
            for j, output in enumerate(outputs):
                # (C, H, W) -> (H, W, C)
                output = output.cpu().numpy().transpose(1, 2, 0)
                
                # 0-1 범위를 0-255 범위로 변환
                output = (output * 255).astype(np.uint8)
                
                # RGB -> BGR (OpenCV format)
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                
                # 이미지 저장
                image_name = f'prediction_{i * test_loader.batch_size + j:05d}.png'
                cv2.imwrite(os.path.join(output_dir, image_name), output)

def main():
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    checkpoint_path = 'checkpoints/best_model.pth'
    output_dir = 'predictions'
    
    # 테스트 데이터셋 및 데이터로더 설정
    test_dataset = ImageDataset('data/test.csv', train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4)
    
    # 모델 로드
    model = UNet().to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 예측 수행 및 저장
    save_predictions(model, test_loader, device, output_dir)
    
    print(f'Predictions saved to {output_dir}')

if __name__ == '__main__':
    main() 