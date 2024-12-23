import os
import platform
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np

from datasets.dataset import ImageDataset
from models.unet import ImprovedUNet
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

def save_sample_images(inputs, outputs, targets, epoch, save_dir='sample_images'):
    """학습 중 샘플 이미지 저장 함수"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 배치에서 최대 4개의 이미지만 선택
    num_images = min(4, inputs.size(0))
    
    # 이미지 그리드 생성
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5*num_images))
    
    def process_image(img_tensor):
        """이미지 처리 함수: float64 시도 후 필요시 float32로 전환"""
        try:
            # 먼저 float64로 시도
            img = img_tensor.cpu().detach().numpy().transpose(1, 2, 0).astype(np.float64)
            
            # 범위를 벗어나는 값이 있는지 확인
            if np.any(img < 0) or np.any(img > 1):
                img = np.clip(img, 0, 1)
                
            return img
            
        except Exception as e:
            print(f"float64 처리 중 오류 발생: {str(e)}")
            print("float32로 전환하여 처리를 시도합니다.")
            
            # float32로 재시도
            img = img_tensor.cpu().detach().numpy().transpose(1, 2, 0).astype(np.float32)
            
            # 범위를 벗어나는 값이 있는지 확인
            if np.any(img < 0) or np.any(img > 1):
                img = np.clip(img, 0, 1)
                
            return img
    
    for i in range(num_images):
        try:
            # 입력 이미지
            input_img = process_image(inputs[i])
            axes[i, 0].imshow(input_img)
            axes[i, 0].set_title('Input')
            axes[i, 0].axis('off')
            
            # 복원된 이미지
            output_img = process_image(outputs[i])
            axes[i, 1].imshow(output_img)
            axes[i, 1].set_title('Restored')
            axes[i, 1].axis('off')
            
            # 타겟(원본) 이미지
            target_img = process_image(targets[i])
            axes[i, 2].imshow(target_img)
            axes[i, 2].set_title('Ground Truth')
            axes[i, 2].axis('off')
            
        except Exception as e:
            print(f"이미지 {i} 처리 중 오류 발생: {str(e)}")
            continue
    
    plt.tight_layout()
    try:
        plt.savefig(f'{save_dir}/epoch_{epoch+1}.png')
    except Exception as e:
        print(f"이미지 저장 중 오류 발생: {str(e)}")
    finally:
        plt.close()

def train(model, train_loader, criterion, optimizer, device, scaler=None, save_images=False, epoch=0):
    model.train()
    total_loss = 0
    
    with tqdm(train_loader, desc='Training') as pbar:
        for batch_idx, (inputs, targets) in enumerate(pbar):
            try:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
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
                
                # 이미지 저장 옵션이 활성화된 경우에만 저장
                if save_images and batch_idx == 0 and (epoch + 1) % 5 == 0:
                    save_sample_images(inputs, outputs, targets, epoch)
                
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

def get_system_info():
    """시스템 사양 정보를 반환합니다."""
    cpu_info = {
        'physical_cores': psutil.cpu_count(logical=False),
        'total_cores': psutil.cpu_count(logical=True),
        'max_frequency': psutil.cpu_freq().max if psutil.cpu_freq() else None,
        'memory_total': round(psutil.virtual_memory().total / (1024.0 ** 3), 2),  # GB
        'memory_available': round(psutil.virtual_memory().available / (1024.0 ** 3), 2)  # GB
    }
    
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'gpu_memory': None
    }
    
    if torch.cuda.is_available():
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            gpu_info['gpu_memory'] = round(gpu_memory, 2)
        except:
            pass
    
    return cpu_info, gpu_info

def estimate_batch_time(device_type, cpu_info, gpu_info, img_size, batch_size):
    """하드웨어 사양을 기반으로 배치당 예상 처리 시간을 계산합니다."""
    base_time = 0.1  # 기본 예상 시간 (초)
    
    if device_type == 'cuda':
        # GPU 성능에 따른 조정
        if gpu_info['device_name']:
            if 'RTX' in gpu_info['device_name']:
                base_time *= 0.3  # RTX 카드는 더 빠름
            elif 'GTX' in gpu_info['device_name']:
                base_time *= 0.5  # GTX 카드
            if gpu_info['gpu_memory'] and gpu_info['gpu_memory'] > 8:
                base_time *= 0.8  # 고용량 GPU 메모리
    
    elif device_type == 'mps':
        # Apple Silicon 기반 예측
        base_time *= 0.6  # M1/M2 칩은 CPU보다 빠름
    
    else:  # CPU
        # CPU 코어 수와 속도에 따른 조정
        if cpu_info['physical_cores'] >= 8:
            base_time *= 0.8
        if cpu_info['max_frequency'] and cpu_info['max_frequency'] > 3000:
            base_time *= 0.9
    
    # 이미지 크기와 배치 크기에 따른 조정
    size_factor = (img_size / 128) ** 2  # 128x128 기준
    batch_factor = (batch_size / 32)  # 배치 크기 32 기준
    
    estimated_time = base_time * size_factor * batch_factor
    return estimated_time

def main():
    # 시스템 정보 출력
    print(f"운영체제: {platform.system()}")
    print(f"Python 버전: {platform.python_version()}")
    print(f"PyTorch 버전: {torch.__version__}")
    
    # 시스템 사양 확인
    cpu_info, gpu_info = get_system_info()
    print("\n시스템 사양:")
    print(f"CPU 코어: {cpu_info['physical_cores']} 물리코어, {cpu_info['total_cores']} 논리코어")
    if cpu_info['max_frequency']:
        print(f"CPU 최대 주파수: {cpu_info['max_frequency']:.2f} MHz")
    print(f"메모리: 전체 {cpu_info['memory_total']}GB, 사용 가능 {cpu_info['memory_available']}GB")
    
    print(f"\nGPU 정보:")
    print(f"CUDA 사용 가능: {gpu_info['cuda_available']}")
    if gpu_info['device_name']:
        print(f"GPU: {gpu_info['device_name']}")
        if gpu_info['gpu_memory']:
            print(f"GPU 메모리: {gpu_info['gpu_memory']}GB")
    
    # 기본 하이퍼파라미터 설정
    num_epochs = 30  # 에폭 수
    learning_rate = 0.001
    
    # 이미지 해상도 선택
    print("\n이미지 해상도 선택:")
    print("1) 128 x 128 (빠른 학습, 낮은 품질)")
    print("2) 256 x 256 (중간 속도, 중간 품질)")
    print("3) 512 x 512 (느린 학습, 원본 품질)")
    
    while True:
        try:
            resolution_choice = int(input("해상도를 선택하세요 (1-3): "))
            if resolution_choice in [1, 2, 3]:
                img_size = {1: 128, 2: 256, 3: 512}[resolution_choice]
                # 해상도에 따른 기본 배치 크기 설정
                suggested_batch_size = {1: 32, 2: 16, 3: 8}[resolution_choice]
                break
            print("1에서 3 사이의 숫자를 입력해주세요.")
        except ValueError:
            print("올바른 숫자를 입력해주세요.")
    
    print(f"\n선택된 이미지 해상도: {img_size} x {img_size}")
    
    # 배치 크기 설정
    print("\n배치 크기 설정:")
    print(f"권장 배치 크기: {suggested_batch_size} (선택한 해상도 {img_size}x{img_size}에 최적화)")
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
    
    # GPU 메모리 요구사항 계산 및 경고
    if gpu_info['cuda_available'] and gpu_info['gpu_memory']:
        # 모델 파라미터와 옵티마이저 상태를 고려한 추가 메모리
        model_memory = 0.5  # 예상 모델 기본 메모리 (GB)
        estimated_memory = (img_size * img_size * 3 * 4 * batch_size * 3) / (1024 * 1024 * 1024) + model_memory  # GB
        print(f"\n예상 GPU 메모리 사용량: {estimated_memory:.1f}GB")
        print(f"사용 가능한 GPU 메모리: {gpu_info['gpu_memory']:.1f}GB")
        
        if estimated_memory > gpu_info['gpu_memory'] * 0.7:  # 70% 이상 사용 시 경고
            print("\n경고: 선택한 설정이 GPU 메모리를 많이 사용할 수 있습니다.")
            print(f"권장 배치 크기: {suggested_batch_size}")
            print("다음 옵션 중 선택해주세요:")
            print("1) 배치 크기 줄이기")
            print("2) 더 낮은 해상도 선택하기")
            print("3) 현재 설정으로 계속하기")
            
            while True:
                choice = input("선택 (1-3): ").strip()
                if choice == "1":
                    while True:
                        try:
                            new_batch = int(input(f"새로운 배치 크기 (권장: {suggested_batch_size} 이하): "))
                            if new_batch > 0:
                                batch_size = new_batch
                                break
                        except ValueError:
                            print("올바른 숫자를 입력해주세요.")
                    break
                elif choice == "2":
                    return main()  # 처음부터 다시 시작
                elif choice == "3":
                    break
                else:
                    print("1-3 사이의 숫자를 입력해주세요.")
    
    print(f"\n최종 설정된 배치 크기: {batch_size}")
    
    # 데이터 증강 여부 선택
    while True:
        aug_input = input("\n데이터 증강을 사용하시겠습니까? (y/n): ").lower()
        if aug_input in ['y', 'n']:
            use_augmentation = (aug_input == 'y')
            break
        print("잘못된 입력입니다. 'y' 또는 'n'을 입력해주세요.")
    
    # 데이터셋 비율 선택
    while True:
        subset_input = input("전체 데이터셋 중 몇 %를 사용하시겠습니까? (1-100): ")
        try:
            subset_fraction = float(subset_input) / 100
            if 0 < subset_fraction <= 1:
                break
            print("1에서 100 사이의 숫자를 입력해주세요.")
        except ValueError:
            print("올바른 숫자를 입력해주세요.")
    
    # 학습 중 이미지 생성 여부 선택
    while True:
        save_images_input = input("학습 중 샘플 이미지를 생성하시겠습니까? (y/n): ").lower()
        if save_images_input in ['y', 'n']:
            save_images = (save_images_input == 'y')
            break
        print("잘못된 입력입니다. 'y' 또는 'n'을 입력해주세요.")
    
    print(f"\n선택된 설정:")
    print(f"- 데이터 증강 사용: {'예' if use_augmentation else '아니오'}")
    print(f"- 사용할 데이터셋 비율: {subset_fraction*100}%")
    print(f"- 샘플 이미지 생성: {'예' if save_images else '아니오'}")
    
    # 디바이스 설정
    device = get_device()
    print(f"Using device: {device}")
    
    # 시스템에 따른 worker 수 설정
    num_workers = get_optimal_num_workers()
    print(f"Number of workers: {num_workers}")
    
    try:
        # 데이터셋 및 데이터로더 설정
        train_dataset = ImageDataset('data/train.csv', train=True, val=False, 
                                   img_size=img_size, use_augmentation=use_augmentation,
                                   subset_fraction=subset_fraction)
        val_dataset = ImageDataset('data/train.csv', train=True, val=True, 
                                 img_size=img_size, use_augmentation=False,
                                 subset_fraction=subset_fraction)
        
        print(f"\n데이터셋 정보:")
        print(f"- 전체 데이터 중 {subset_fraction*100}% 사용")
        print(f"- 학습 데이터 크기: {len(train_dataset)}")
        print(f"- 검증 데이터 크기: {len(val_dataset)}")
        
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
        batch_time = estimate_batch_time(
            device.type, cpu_info, gpu_info,
            img_size=img_size, batch_size=batch_size
        )
        estimated_time_per_epoch = len(train_loader) * batch_time
        estimated_total_time = estimated_time_per_epoch * num_epochs
        
        print(f"\n학습 시간 예측:")
        print(f"- 배치당 예상 시간: {batch_time:.3f}초")
        print(f"- 에폭당 예상 시간: {estimated_time_per_epoch/60:.1f}분")
        print(f"- 전체 예상 시간: {estimated_total_time/3600:.1f}시간 ({estimated_total_time/60:.1f}분)")
        
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
        model = ImprovedUNet().to(device)
        if device.type == 'cuda' and torch.__version__ >= "2.0.0":
            try:
                model = torch.compile(model, mode='reduce-overhead')
                #model = torch.compile(model, mode='default')
                print("Model compilation successful")
            except Exception as e:
                print(f"Model compilation failed: {str(e)}")
        
        # 메모리 사용량 최적화를 위한 torch.cuda.empty_cache() 호출
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        criterion = CombinedLoss().to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Mixed Precision Training을 위한 scaler (CUDA only)
        scaler = GradScaler() if device.type == 'cuda' else None
        
        # 체크포인트 디렉토리 생성
        os.makedirs('checkpoints', exist_ok=True)
        
        # 학습 기록
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        # 이미지 저장 디렉토리 생성 (이미지 저장 옵션이 선택된 경우에만)
        if save_images:
            os.makedirs('sample_images', exist_ok=True)
            print("\n학습 중 생성된 이미지는 'sample_images' 디렉토리에 저장됩니다.")
            print("5 에폭마다 샘플 이미지가 저장됩니다.")
        
        # 학습 루프
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            
            train_loss = train(model, train_loader, criterion, optimizer, device, scaler, save_images, epoch)
            val_loss = validate(model, val_loader, criterion, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            
            # 학습률 조정
            scheduler.step()
            
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
