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
from models.colorization.pix2pix import Pix2Pix

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
    elif platform.system() == 'Darwin':  # macOS
        return 0  # Mac에서도 일단 0으로 설정
    else:
        return min(os.cpu_count(), 4)  # Linux 등에서는 CPU 코어 수와 4 중 작은 값 사용

def save_sample_images(inputs, outputs, targets, epoch, save_dir='sample_images'):
    """학습 중 샘플 이미지 저장 함수"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 배치에서 최대 4개의 이미지만 선택
    num_images = min(4, inputs.size(0))
    
    # 이미지 그리드 생성
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5*num_images))
    
    for i in range(num_images):
        try:
            # 입력 이미지 (흑백)
            input_img = inputs[i].cpu().detach().numpy().transpose(1, 2, 0)
            input_img = np.repeat(input_img, 3, axis=2)  # 흑백을 3채널로 변환
            axes[i, 0].imshow(input_img)
            axes[i, 0].set_title('Grayscale Input')
            axes[i, 0].axis('off')
            
            # 생성된 컬러 이미지
            output_img = outputs[i].cpu().detach().numpy().transpose(1, 2, 0)
            output_img = np.clip(output_img, 0, 1)
            axes[i, 1].imshow(output_img)
            axes[i, 1].set_title('Generated Color')
            axes[i, 1].axis('off')
            
            # 실제 컬러 이미지
            target_img = targets[i].cpu().detach().numpy().transpose(1, 2, 0)
            axes[i, 2].imshow(target_img)
            axes[i, 2].set_title('Ground Truth')
            axes[i, 2].axis('off')
            
        except Exception as e:
            print(f"이미지 {i} 처리 중 오류 발생: {str(e)}")
            continue
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/epoch_{epoch+1}.png')
    plt.close()

def train(model, train_loader, device, scaler=None, save_images=False, epoch=0):
    model.train()
    total_gen_loss = 0
    total_disc_loss = 0
    
    with tqdm(train_loader, desc='Training') as pbar:
        for batch_idx, (gray_images, color_images) in enumerate(pbar):
            try:
                gray_images = gray_images.to(device, non_blocking=True)
                color_images = color_images.to(device, non_blocking=True)
                
                # Convert RGB to Grayscale if needed
                if gray_images.shape[1] == 3:
                    gray_images = 0.299 * gray_images[:, 0] + 0.587 * gray_images[:, 1] + 0.114 * gray_images[:, 2]
                    gray_images = gray_images.unsqueeze(1)
                
                if device.type == 'cuda':
                    with autocast():
                        losses = model.train_step(gray_images, color_images)
                else:
                    losses = model.train_step(gray_images, color_images)
                
                total_gen_loss += losses['gen_loss']
                total_disc_loss += losses['disc_loss']
                
                pbar.set_postfix({
                    'gen_loss': losses['gen_loss'],
                    'disc_loss': losses['disc_loss']
                })
                
                # 이미지 저장 옵션이 활성화된 경우에만 저장
                if save_images and batch_idx == 0 and (epoch + 1) % 5 == 0:
                    with torch.no_grad():
                        fake_color = model.generate(gray_images)
                        save_sample_images(gray_images, fake_color, color_images, epoch)
                
            except RuntimeError as e:
                print(f"Error during training: {str(e)}")
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("GPU 메모리 부족. 배치를 건너뜁니다.")
                    continue
                else:
                    raise e
    
    return {
        'gen_loss': total_gen_loss / len(train_loader),
        'disc_loss': total_disc_loss / len(train_loader)
    }

def validate(model, val_loader, device):
    model.eval()
    total_gen_loss = 0
    total_disc_loss = 0
    
    with torch.no_grad():
        with tqdm(val_loader, desc='Validation') as pbar:
            for gray_images, color_images in pbar:
                try:
                    gray_images = gray_images.to(device, non_blocking=True)
                    color_images = color_images.to(device, non_blocking=True)
                    
                    # Convert RGB to Grayscale if needed
                    if gray_images.shape[1] == 3:
                        gray_images = 0.299 * gray_images[:, 0] + 0.587 * gray_images[:, 1] + 0.114 * gray_images[:, 2]
                        gray_images = gray_images.unsqueeze(1)
                    
                    fake_color = model.generate(gray_images)
                    disc_real = model.discriminator(gray_images, color_images)
                    disc_fake = model.discriminator(gray_images, fake_color)
                    
                    # Calculate losses
                    disc_real_loss = model.bce_loss(disc_real, torch.ones_like(disc_real))
                    disc_fake_loss = model.bce_loss(disc_fake, torch.zeros_like(disc_fake))
                    disc_loss = (disc_real_loss + disc_fake_loss) / 2
                    
                    gen_adversarial_loss = model.bce_loss(disc_fake, torch.ones_like(disc_fake))
                    gen_l1_loss = model.l1_loss(fake_color, color_images) * model.lambda_l1
                    gen_loss = gen_adversarial_loss + gen_l1_loss
                    
                    total_gen_loss += gen_loss.item()
                    total_disc_loss += disc_loss.item()
                    
                    pbar.set_postfix({
                        'gen_loss': gen_loss.item(),
                        'disc_loss': disc_loss.item()
                    })
                    
                except RuntimeError as e:
                    print(f"Error during validation: {str(e)}")
                    if "out of memory" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print("GPU 메모리 부족. 배치를 건너뜁니다.")
                        continue
                    else:
                        raise e
    
    return {
        'gen_loss': total_gen_loss / len(val_loader),
        'disc_loss': total_disc_loss / len(val_loader)
    }

def plot_losses(gen_losses, disc_losses, val_gen_losses, val_disc_losses):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(gen_losses, label='Train Generator Loss')
    plt.plot(val_gen_losses, label='Val Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Generator Losses')
    
    plt.subplot(1, 2, 2)
    plt.plot(disc_losses, label='Train Discriminator Loss')
    plt.plot(val_disc_losses, label='Val Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Discriminator Losses')
    
    plt.tight_layout()
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
            pin_memory=True if device.type == 'cuda' else False,  # CUDA에서만 pin_memory 사용
            persistent_workers=False,  # worker 관련 문제 해결을 위해 비활성화
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False,  # CUDA에서만 pin_memory 사용
            persistent_workers=False,  # worker 관련 문제 해결을 위해 비활성화
            drop_last=True
        )
        
        print(f"배치 크기: {batch_size}")
        print(f"총 배치 수 (학습): {len(train_loader)}")
        print(f"총 배치 수 (검증): {len(val_loader)}")
        print(f"총 에폭 수: {num_epochs}")
        
        # 예크포인트 확인 및 모델 초기화
        checkpoint_path = 'checkpoints/interrupted_model.pth'
        start_epoch = 0
        gen_losses = []
        disc_losses = []
        val_gen_losses = []
        val_disc_losses = []
        
        model = Pix2Pix(lr=learning_rate).to(device)
        
        if os.path.exists(checkpoint_path):
            print(f"\n중단된 학습 체크포인트를 발견했습니다: {checkpoint_path}")
            while True:
                resume = input("중단된 학습을 이어서 하시겠습니까? (y/n): ").lower()
                if resume in ['y', 'n']:
                    break
                print("잘못된 입력입니다. 'y' 또는 'n'을 입력해주세요.")
            
            if resume == 'y':
                try:
                    print("체크포인트를 로드합니다...")
                    model.load_checkpoint(checkpoint_path, device)
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    start_epoch = checkpoint.get('epoch', 0)
                    gen_losses = checkpoint.get('gen_losses', [])
                    disc_losses = checkpoint.get('disc_losses', [])
                    val_gen_losses = checkpoint.get('val_gen_losses', [])
                    val_disc_losses = checkpoint.get('val_disc_losses', [])
                    
                    print(f"체크포인트 로드 완료. {start_epoch}번째 에폭부터 학습을 재개합니다.")
                    
                except Exception as e:
                    print(f"체크포인트 로드 중 오류 발생: {str(e)}")
                    print("새로운 학습을 시작합니다.")
                    model = Pix2Pix(lr=learning_rate).to(device)
        
        # 학습 루프
        for epoch in range(start_epoch, num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            
            train_losses = train(model, train_loader, device, save_images, epoch)
            val_losses = validate(model, val_loader, device)
            
            gen_losses.append(train_losses['gen_loss'])
            disc_losses.append(train_losses['disc_loss'])
            val_gen_losses.append(val_losses['gen_loss'])
            val_disc_losses.append(val_losses['disc_loss'])
            
            print(f'Training Generator Loss: {train_losses["gen_loss"]:.4f}')
            print(f'Training Discriminator Loss: {train_losses["disc_loss"]:.4f}')
            print(f'Validation Generator Loss: {val_losses["gen_loss"]:.4f}')
            print(f'Validation Discriminator Loss: {val_losses["disc_loss"]:.4f}')
            
            # 중간 체크포인트 저장
            try:
                checkpoint = {
                    'epoch': epoch + 1,
                    'gen_losses': gen_losses,
                    'disc_losses': disc_losses,
                    'val_gen_losses': val_gen_losses,
                    'val_disc_losses': val_disc_losses,
                }
                model.save_checkpoint('checkpoints/interrupted_model.pth')
                torch.save(checkpoint, 'checkpoints/training_state.pth')
            except Exception as e:
                print(f"Error saving checkpoint: {str(e)}")
            
            # 매 10 에폭마다 체크포인트와 손실 그래프 저장
            if (epoch + 1) % 10 == 0:
                try:
                    model.save_checkpoint(f'checkpoints/model_epoch_{epoch+1}.pth')
                    plot_losses(gen_losses, disc_losses, val_gen_losses, val_disc_losses)
                    print(f'Saved checkpoint at epoch {epoch+1}')
                except Exception as e:
                    print(f"Error saving checkpoint: {str(e)}")
        
    except KeyboardInterrupt:
        print("\n학습이 사용자에 의해 중단되었습니다.")
        try:
            model.save_checkpoint('checkpoints/interrupted_model.pth')
            print("중단된 모델이 저장되었습니다.")
        except Exception as e:
            print(f"Error saving interrupted model: {str(e)}")
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e

if __name__ == '__main__':
    main() 
