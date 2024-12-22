import os
import platform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets.dataset import ImageDataset
from models.unet import UNet
from models.losses import CombinedLoss

def get_device():
    """시스템 환경에 따른 적절한 디바이스 반환"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_optimal_num_workers():
    """시스템에 따른 최적의 worker 수 반환"""
    if platform.system() == 'Windows':
        return 0  # Windows에서는 0이 더 안정적
    else:
        return min(os.cpu_count(), 4)  # CPU 코어 수와 4 중 작은 값 사용

def train(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    
    with tqdm(train_loader, desc='Training') as pbar:
        for inputs, targets in pbar:
            try:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                # Mixed Precision Training (CUDA only)
                if device.type == 'cuda':
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
            except RuntimeError as e:
                print(f"Error during training: {str(e)}")
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("GPU 메모리 부족. 배치를 건너뜁니다.")
                    continue
                else:
                    raise e
            
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        with tqdm(val_loader, desc='Validation') as pbar:
            for inputs, targets in pbar:
                try:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    
                    if device.type == 'cuda':
                        with autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
                    
                except RuntimeError as e:
                    print(f"Error during validation: {str(e)}")
                    if "out of memory" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print("GPU 메모리 부족. 배치를 건너뜁니다.")
                        continue
                    else:
                        raise e
    
    return total_loss / len(val_loader)

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()

def main():
    # 시스템 정보 출력
    print(f"운영체제: {platform.system()}")
    print(f"Python 버전: {platform.python_version()}")
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"사용 가능한 GPU: {torch.cuda.get_device_name(0)}")
    
    # 디이터 증강 여부 선택
    while True:
        aug_input = input("데이터 증강을 사용하시겠습니까? (y/n): ").lower()
        if aug_input in ['y', 'n']:
            use_augmentation = (aug_input == 'y')
            break
        print("잘못된 입력입니다. 'y' 또는 'n'을 입력해주세요.")
    
    print(f"데이터 증강 사용: {'예' if use_augmentation else '아니오'}")
    
    # 디바이스 설정
    device = get_device()
    print(f"Using device: {device}")
    
    # 시스템에 따른 worker 수 설정
    num_workers = get_optimal_num_workers()
    print(f"Number of workers: {num_workers}")
    
    # 하이퍼파라미터 설정
    batch_size = 32  # 배치 크기
    num_epochs = 30  # 에폭 수 감소
    learning_rate = 0.001
    
    try:
        # 데이터셋 및 데이터로더 설정
        train_dataset = ImageDataset('data/train.csv', train=True, val=False, 
                                   img_size=128, use_augmentation=use_augmentation)
        val_dataset = ImageDataset('data/train.csv', train=True, val=True, 
                                 img_size=128, use_augmentation=False)
        
        print(f"학습 데이터 크기: {len(train_dataset)}")
        print(f"검증 데이터 크기: {len(val_dataset)}")
        
        # 데이터 로딩 최적화
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if device.type != 'cpu' else False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device.type != 'cpu' else False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True
        )
        
        print(f"배치 크기: {batch_size}")
        print(f"총 배치 수 (학습): {len(train_loader)}")
        print(f"총 배치 수 (검증): {len(val_loader)}")
        print(f"총 에폭 수: {num_epochs}")
        
        # 예상 학습 시간 계산
        estimated_time_per_epoch = len(train_loader) * 0.1  # 배치당 약 0.1초로 가정
        estimated_total_time = estimated_time_per_epoch * num_epochs
        print(f"예상 학습 시간: {estimated_total_time/60:.1f}분 (배치당 0.1초 가정)")
        
        # 학습 시작 확인
        while True:
            confirm = input("\n이대로 학습을 시작하시겠습니까? (y/n): ").lower()
            if confirm in ['y', 'n']:
                if confirm == 'n':
                    print("학습이 취소되었습니다.")
                    return
                break
            print("잘못된 입력입니다. 'y' 또는 'n'을 입력해주세요.")
        
        # 모델 설정
        model = UNet().to(device)
        if device.type == 'cuda' and torch.__version__ >= "2.0.0":
            try:
                model = torch.compile(model, mode='reduce-overhead')  # 컴파일 모드 최적화
                print("Model compilation successful")
            except Exception as e:
                print(f"Model compilation failed: {str(e)}")
        
        # 메모리 사용량 최적화를 위한 torch.cuda.empty_cache() 호출
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        criterion = CombinedLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        # Mixed Precision Training을 위한 scaler (CUDA only)
        scaler = GradScaler() if device.type == 'cuda' else None
        
        # 체크포인트 디렉토리 생성
        os.makedirs('checkpoints', exist_ok=True)
        
        # 학습 기록
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        # 학습 루프
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            
            train_loss = train(model, train_loader, criterion, optimizer, device, scaler)
            val_loss = validate(model, val_loader, criterion, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            
            # 학습률 조정
            scheduler.step(val_loss)
            
            # 체크포인트 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                try:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, 'checkpoints/best_model.pth')
                    print(f'Saved best model with validation loss: {val_loss:.4f}')
                except Exception as e:
                    print(f"Error saving checkpoint: {str(e)}")
            
            # 매 10 에폭마다 체크포인트 저장
            if (epoch + 1) % 10 == 0:
                try:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, f'checkpoints/model_epoch_{epoch+1}.pth')
                    print(f'Saved checkpoint at epoch {epoch+1}')
                    
                    # 손실 그래프 저장
                    plot_losses(train_losses, val_losses)
                except Exception as e:
                    print(f"Error saving checkpoint: {str(e)}")
                    
    except KeyboardInterrupt:
        print("\n학습이 사용자에 의해 중단되었습니다.")
        # 마지막 체크포인트 저장
        try:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'checkpoints/interrupted_model.pth')
            print("중단된 모델이 저장되었습니다.")
        except Exception as e:
            print(f"Error saving interrupted model: {str(e)}")
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e

if __name__ == '__main__':
    main() 