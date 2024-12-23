import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as np
import platform

from datasets.dataset import ImageDataset
from models.unet import ImprovedUNet

def get_optimal_num_workers():
    """시스템에 따른 최적의 worker 수 반환"""
    if platform.system() == 'Windows':
        return 0  # Windows에서는 0이 더 안정적
    else:
        return min(os.cpu_count(), 4)  # CPU 코어 수와 4 중 작은 값 사용

def get_device():
    """시스템 환경에 따른 적절한 디바이스 반환"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def save_predictions(model, test_loader, device, output_dir):
    """테스트 데이터에 대한 예측을 수행하고 결과를 저장합니다."""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, inputs in enumerate(tqdm(test_loader, desc='이미지 생성 중')):
            try:
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
                    
            except Exception as e:
                print(f"이미지 {i * test_loader.batch_size + j} 처리 중 오류 발생: {str(e)}")
                continue

def main():
    # 이미지 해상도 선택
    print("\n이미지 해상도 선택:")
    print("1) 128 x 128 (빠른 처리)")
    print("2) 256 x 256 (중간 품질)")
    print("3) 512 x 512 (최고 품질)")
    
    while True:
        try:
            resolution_choice = int(input("해상도를 선택하세요 (1-3): "))
            if resolution_choice in [1, 2, 3]:
                img_size = {1: 128, 2: 256, 3: 512}[resolution_choice]
                break
            print("1에서 3 사이의 숫자를 입력해주세요.")
        except ValueError:
            print("올바른 숫자를 입력해주세요.")
    
    # 배치 크기 설정
    suggested_batch_size = {1: 32, 2: 16, 3: 8}[resolution_choice]
    print(f"\n권장 배치 크기: {suggested_batch_size}")
    while True:
        try:
            batch_input = input(f"배치 크기를 입력하세요 (기본값: {suggested_batch_size}): ").strip()
            if batch_input == "":
                batch_size = suggested_batch_size
                break
            batch_size = int(batch_input)
            if batch_size > 0:
                break
            print("배치 크기는 양수여야 합니다.")
        except ValueError:
            print("올바른 숫자를 입력해주세요.")
    
    # 설정
    device = get_device()
    print(f"\n사용 중인 디바이스: {device}")
    
    num_workers = get_optimal_num_workers()
    print(f"Worker 수: {num_workers}")
    
    checkpoint_path = 'checkpoints/best_model.pth'
    output_dir = 'predictions'
    
    # 테스트 데이터셋 및 데이터로더 설정
    test_dataset = ImageDataset('data/test.csv', train=False, img_size=img_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type != 'cpu' else False
    )
    
    print(f"\n데이터셋 정보:")
    print(f"- 테스트 데이터 크기: {len(test_dataset)}")
    print(f"- 치 크기: {batch_size}")
    print(f"- 총 배치 수: {len(test_loader)}")
    
    try:
        # 모델 로드
        print("\n모델 로드 중...")
        model = ImprovedUNet().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("모델 로드 완료")
        
        # 예측 수행 및 저장
        print(f"\n결과물은 '{output_dir}' 디렉토리에 저장됩니다.")
        save_predictions(model, test_loader, device, output_dir)
        
        print(f'\n추론 완료! 결과물이 {output_dir} 디렉토리에 저장되었습니다.')
        
    except FileNotFoundError:
        print(f"\n오류: {checkpoint_path} 파일을 찾을 수 없습니다.")
        print("먼저 모델을 학습시켜주세요.")
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")

if __name__ == '__main__':
    main() 