# 이미지 색상화 및 복원 AI

흑백 이미지의 색상화 및 손실된 부분을 복원하는 AI 모델
- 이론적 학습 데이터 수
     - 색상화, 손실 : 각 29602개
## Unet 모델 설명
- https://wikidocs.net/148870

## 데이터셋 구조 설명
### 데이터셋 디렉토리 (data/)
1. train_input/ 
   - 학습용 입력 이미지 디렉토리
   - 흑백 또는 손상된 이미지들이 포함
   - 파일 형식: PNG (TRAIN_XXXXX.png)
   - 이미지 크기: 512x512
   - 특징: 
     * 색상 정보가 제거된 흑백 이미지
     * 일부 영역이 손상된 이미지
     * 노이즈가 추가된 이미지

2. train_gt/ (Ground Truth)
   - 학습용 정답 이미지 디렉토리
   - 원본 컬러 이미지들이 포함
   - 파일 형식: PNG (TRAIN_XXXXX.png)
   - 이미지 크기: 512x512
   - 특징:
     * 완벽한 상태의 컬러 이미지
     * 손상되지 않은 깨끗한 이미지
     * train_input의 각 이미지와 1:1 매칭

3. test_input/
   - 테스트용 입력 이미지 디렉토리
   - 실제 모델 평가에 사용되는 이미지들

4. CSV 파일
   - train.csv: 학습 데이터 경로 정보
     * input_image_path: train_input 이미지 경로
     * gt_image_path: train_gt 이미지 경로
   - test.csv: 테스트 데이터 경로 정보
     * input_image_path: test_input 이미지 경로

## 프로젝트 구조 
### 이용시 구조는 무조건 같아야합니다.
```
.
├── data/               # 데이터셋 디렉토리
│   ├── train_input/   # 학습용 입력 이미지
│   ├── train_gt/      # 학습용 정답 이미지
│   ├── test_input/    # 테스트용 입력 이미지
│   ├── train.csv      # 학습 데이터 경로 정보
│   └── test.csv       # 테스트 데이터 경로 정보
│
├── src/               # 소스 코드
│   ├── models/        # 모델 관련 코드
│   │   ├── unet.py   # U-Net 모델 구현 
│   │   └── losses.py # 손실 함수 구현
│   ├── datasets/      # 데이터셋 처리
│   │   └── dataset.py # 데이터 로딩 및 전처리
│   ├── train.py      # 학습 실행 코드
│   └── inference.py  # 추론 실행 코드
│
├── checkpoints/       # 모델 체크포인트 저장
└── requirements.txt   # 필요한 패키지 목록
```
## 데이터 준비

1. **코드 다운로드**
```bash
git clone https://github.com/bunhine0452/dacon236420.git
cd dacon236420
```

2. **데이터셋 다운로드**
- [데이콘 경진대회 페이지](https://dacon.io/competitions/official/236420/data)에서 데이터셋 다운로드

3. **데이터 배치**
- 프로젝트 루트에 `data` 디렉토리 생성
```bash
mkdir data
```
- 다운로드 받은 데이터셋의 압축을 풀어 `data` 디렉토리 내에 배치
- 최종 디렉토리 구조:
```
data/
├── train_input/   # 학습용 입력 이미지
├── train_gt/      # 학습용 정답 이미지
├── test_input/    # 테스트용 입력 이미지
├── train.csv      # 학습 데이터 경로 정보
└── test.csv       # 테스트 데이터 경로 정보
```

4. **데이터 구조 확인**
- train.csv와 test.csv 파일이 올바른 경로 정보를 포함하고 있는지 확인
- 각 이미지 파일이 지정된 디렉토리에 올바르게 위치하는지 확인

## 코드 구조 설명

### 1. 데이터셋 처리 (src/datasets/dataset.py)
- `ImageDataset` 클래스: PyTorch Dataset 구현
- 주요 기능:
  - 이미지 로딩 및 전처리
  - 데이터 증강 (선택적 적용)
  - Train/Validation 분할 (8:2 비율)
- 수정 방법:
  - 이미지 크기: `img_size` 파라미터 변경
  - 데이터 증강: `Compose` 내의 변환 수정
  - 분할 비율: `sample` 함수의 `frac` 값 수정

### 2. 모델 구현 (src/models/)

#### U-Net 모델 (unet.py)
- 기본적인 U-Net 아키텍처 구현
- 인코더-디코더 구조
- Skip Connection 사용
- 수정 방법:
  - 레이어 수정: `UNet` 클래스 내 레이어 구성 변경
  - 채널 수 변경: `in_channels`, `out_channels` 파라미터 수정

#### 손실 함수 (losses.py)
- `VGGPerceptualLoss`: VGG16 기반 지각적 손실
- `CombinedLoss`: L1 손실과 지각적 손실 결합
- 수정 방법:
  - 손실 가중치: `lambda_perceptual`, `lambda_l1` 값 조정
  - VGG 레이어: `slice1`, `slice2` 등의 레이어 구성 변경

### 3. 학습 코드 (src/train.py)
- 주요 기능:
  - 데이터 로딩 및 학습 설정
  - Mixed Precision Training
  - 체크포인트 저장
  - 학습 모니터링
- 수정 방법:
  - 하이퍼파라미터: `batch_size`, `num_epochs`, `learning_rate` 등 수정
  - 옵티마이저: `optimizer` 설정 변경
  - 학습률 스케줄러: `scheduler` 설정 변경

### 4. 추론 코드 (src/inference.py)
- 학습된 모델을 사용한 이미지 생성
- 결과 저장 및 후처리
- 수정 방법:
  - 배치 크기: `batch_size` 수정
  - 출력 형식: `save_predictions` 함수 수정

## 설치 방법

1. 환경 설정
```bash
# Conda 환경 생성
conda create -n image python=3.10
conda activate image

# 필요한 패키지 설치
pip install -r requirements.txt
# 디코딩 에러시 이용
pip install -r requirements_ansi.txt

# windows 경우 cuda 버전에 알맞는 torch 버전 설치

pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 이 명령어는 제 pc환경 cuda 버전:12.7에 사용했습니다.
# cuda는 하위 버전 호환 가능 12.7 > cu121(cuda12.1)
```

2. 데이터 준비
- `data` 디렉토리에 데이터셋 구성
- CSV 파일 형식 확인 (input_image_path, gt_image_path 컬럼 필요)

## 사용 방법

### 1. 학습 실행
```bash
cd dacon236420
python src/train.py  #src 파일 내에서 실행이 아니라 dacon236420파일에서 실행됩니다. cd src 하거나 그냥 train.py를 직접 실행시킬경우 경로 오류 발생 
```

실행 시 다음과 같은 정보와 선택 사항이 순차적으로 제공됩니다:

1. **시스템 정보 출력**
   - 운영체제 및 Python/PyTorch 버전
   - CPU 정보 (코어 수, 주파수)
   - 메모리 정보 (전체/사용 가능)
   - GPU 정보 (사용 가능 여부, 모델명, 메모리)

2. **이미지 해상도 선택**
   - 128 x 128 (빠른 학습, 낮은 품질)
   - 256 x 256 (중간 속도, 중간 품질)
   - 512 x 512 (느린 학습, 원본 품질)

3. **배치 크기 설정**
   - 해상도별 권장 배치 크기 제공
   - GPU 메모리 사용량 계산 및 경고
   - 메모리 부족 시 대안 제시:
     * 배치 크기 줄이기
     * 더 낮은 해상도 선택
     * 현재 설정 유지

4. **학습 설정 선택**
   - 데이터 증강 사용 여부 (y/n)
     * y: 데이터 증강 기법 적용
     * n: 원본 이미지만 사용
   
   - 데이터셋 사용 비율 선택 (1-100%)
     * 전체 데이터셋 중 사용할 비율을 지정
     * 예: 30 입력 시 전체 데이터의 30%만 사용
     * 학습 시간 단축을 위해 작은 비율로 시작하는 것을 추천
   
   - 학습 중 샘플 이미지 생성 여부 (y/n)
     * y: 5 에폭마다 복원 결과 이미지 저장
     * n: 이미지 저장하지 않음

5. **학습 정보 확인**
   - 선택된 설정 요약
   - 데이터셋 크기 정보
   - 학습/검증 데이터 크기
   - 배치 크기 및 총 배치 수
   - 총 에폭 수

6. **예상 학습 시간**
   - 배치당 예상 처리 시간
   - 에폭당 예상 소요 시간
   - 전체 학습 예상 시간

7. **최종 확인**
   - 위 설정으로 학습 시작 여부 확인 (y/n)

💡 **선택 가이드**:
- **해상도 선택**:
  * 빠른 테스트: 128 x 128
  * 일반 학습: 256 x 256
  * 고품질: 512 x 512

- **배치 크기**:
  * 128 x 128: 32개 권장
  * 256 x 256: 16개 권장
  * 512 x 512: 8개 권장

- **데이터셋 비율**:
  * 빠른 테스트: 10-20% 
  * 기본 학습: 30-50% 
  * 완전한 학습: 100% 

### 2. 학습 결과물
- `checkpoints/`: 모델 가중치 저장
  * `best_model.pth`: 최상의 성능 모델
  * `model_epoch_N.pth`: 10 에폭마다 저장되는 체크포인트
  * `interrupted_model.pth`: 학습 중단 시 저장되는 모델

- `sample_images/`: 학습 중 생성된 이미지 (선택 시)
  * 5 에폭마다 저장
  * 각 이미지는 입력/복원/원본 비교 가능
  * 최대 4개의 샘플 이미지 표시

- `loss_plot.png`: 학습/검증 손실 그래프
  * 10 에폭마다 자동 업데이트
  * 학습 진행 상황 시각화

### 3. 학습 중단 및 재개
- 학습 중 `Ctrl+C`로 안전하게 중단 가능
- 중단 시 자동으로 `interrupted_model.pth` 저장
- 체크포인트를 통해 학습 재개 가능

### 4. 시스템 요구사항
- **CPU**: 멀티코어 프로세서 권장
- **RAM**: 최소 8GB, 16GB 이상 권장
- **GPU**: 
  * NVIDIA GPU: CUDA 지원 필수
  * Apple Silicon: MPS 가속 지원
- **저장공간**: 최소 10GB 여유공간

### 5. 문제 해결
- GPU 메모리 부족 시:
  * 배치 크기 감소
  * 이미지 해상도 감소
  * 데이터셋 비율 감소

- 학습이 너무 느릴 경우:
  * 더 작은 해상도 선택
  * 데이터셋 비율 감소
  * 데이터 증강 비활성화

## 데이터 증강 (Data Augmentation)

### 구현된 데이터 증강 기법
1. **기하학적 변환**
   - `HorizontalFlip`: 좌우 반전 (p=0.5)
   - `VerticalFlip`: 상하 반전 (p=0.5)
   - `RandomRotate90`: 90도 회전 (p=0.5)

2. **색상 및 밝기 변환**
   - `RandomBrightnessContrast`: 
     * 밝기 변화: ±20%
     * 대비 변화: ±20%
     * 적용 확률: 50%
   - `ColorJitter`:
     * 밝기, 대비, 채도: ±20%
     * 색조: ±10%
     * 적용 확률: 30%

3. **노이즈 및 블러**
   - `GaussianBlur`: 
     * 블러 커널 크기: 3~7
     * 적용 확률: 30%
   - `GaussNoise`:
     * 노이즈 강도: 10.0~50.0
     * 적용 확률: 30%

### 데이터 증강 사용 방법
- 학습 시작 시 데이터 증강 사용 여부 선택 가능
- 검증 데이터에는 증강이 적용되지 않음
- 각 증강 기법은 지정된 확률로 독립적으로 적용

### 데이터 증강 최적화 방법
1. **증강 강도 조절**
   - 각 변환의 범위를 조절하여 적절한 강도 설정
   - 너무 강한 변환은 오히려 성능을 저하시킬 수 있음

2. **적용 확률 조정**
   - 각 증강 기법의 적용 확률(p)을 조절
   - 데이터셋 특성에 따라 최적의 확률 설정

3. **선택적 적용**
   - 특정 증강 기법을 활성화/비활성화
   - 데이터셋과 태스크에 적합한 증강 기법 선택

## U-Net 모델 구조

### 구현된 U-Net 구조
1. **인코더 (Encoder)**
   - 4단계의 다운샘플링
   - 각 단계:
     * Double Convolution (3x3 conv + BN + ReLU)
     * MaxPooling (2x2)
   - 채널 크기: 64 → 128 → 256 → 512

2. **병목 (Bottleneck)**
   - Double Convolution
   - 채널 크기: 512 → 1024

3. **디코더 (Decoder)**
   - 4단계의 업샘플링
   - 각 단계:
     * Transposed Convolution (2x2)
     * Skip Connection으로 인코더 특징과 결합
     * Double Convolution
   - 채널 크기: 1024 → 512 → 256 → 128 → 64

4. **출력 레이어**
   - 1x1 Convolution
   - Sigmoid 활성화 함수
   - 출력: 3채널 RGB 이미지

### Skip Connection
- 인코더의 특징을 디코더로 직접 전달
- 고해상도 특징 보존
- 그래디언트 소실 문제 완화

### 활성화 함수
- 내부: ReLU (Rectified Linear Unit)
- 출력: Sigmoid (0-1 범위의 픽셀값)

### 정규화
- Batch Normalization: 각 합성곱 레이어 후 적용
- 학습 안정화 및 속도 향상

### 개선 가능한 부분
1. **구조 개선**
   - Attention 메커니즘 추가
   - Residual Connection 도입
   - 더 깊은 네트워크 구조

2. **활성화 함수**
   - LeakyReLU 사용
   - ELU 실험
   - PReLU 적용

3. **정규화 기법**
   - Instance Normalization
   - Layer Normalization
   - Group Normalization

4. **손실 함수**
   - Perceptual Loss 가중치 조정
   - Style Loss 추가
   - Total Variation Loss 도입

## 개선된 U-Net 모델 (Improved U-Net)

### 주요 개선사항

1. **Attention 메커니즘**
   - Attention Gate 추가로 중요 특징에 집중
   - 각 디코더 레벨에 적용되어 세부 정보 보존 강화
   - 공간적 주의 집중으로 더 정확한 복원 가능

2. **Squeeze-and-Excitation (SE) 블록**
   - 채널 간 상호작용 모델링
   - 중요 특징 맵 강조
   - 각 DoubleConv 블록 후에 적용

3. **Residual Connection**
   - 그래디언트 소실 문제 완화
   - 더 깊은 네트워크 학습 가능
   - 각 레벨에 residual block 추가

4. **향상된 정규화 및 활성화**
   - BatchNorm → InstanceNorm 변경
   - ReLU → LeakyReLU (0.2) 사용
   - 더 안정적인 학습과 성능 향상

### 최적화된 학습 전략

1. **개선된 옵티마이저**
   - AdamW 사용 (weight decay 포함)
   - L2 정규화로 과적합 방지
   - 가중치 감쇠: 0.01

2. **고급 학습률 스케줄링**
   - Cosine Annealing with Warm Restarts
   - 주기적인 학습률 재시작
   - 더 나은 지역 최적해 탈출

3. **손실 함수 조합**
   - L1 Loss: 픽셀 단위 복원
   - Perceptual Loss: 고수준 특징 보존
   - Style Loss: 텍스처 품질 향상

### 성능 향상 포인트

1. **이미지 품질**
   - 더 선명한 경계선
   - 자연스러운 색상 복원
   - 세부 디테일 보존

2. **학습 안정성**
   - 그래디언트 소실 감소
   - 더 안정적인 학습 곡선
   - 수렴 속도 향상

3. **메모리 효율성**
   - Instance Normalization 사용으로 배치 크기 의존성 감소
   - 최적화된 네트워크 구조
   - 효율적인 attention 연산

### 하이퍼파라미터 가이드

1. **학습률 설정**
   - 초기 학습률: 0.001
   - 최소 학습률: 1e-6
   - Warm Restart 주기: 10 에폭

2. **정규화 강도**
   - Weight Decay: 0.01
   - Dropout: 사용하지 않음 (Instance Norm과 SE 블록이 충분한 정규화 제공)

3. **배치 크기**
   - GPU 메모리에 따라 32-64 권장
   - Instance Normalization으로 작은 배치에서도 안정적
  




# 뜰수있는 오류 및 확인사항들
- windows 환경 pytorch gpu 사용할 때 cuda 버전에 맞는 pytorch 버전 재설치 : 현재 requirement 는 cpu 버전을 설치하고 있습니다.
- windows 실행 오류 시 train.py 수정.
  ```python
   # line 454
   model = torch.compile(model, mode='reduce-overhead') # 이부분을

   # <수정>
   model = torch.compile(model, mode='default')

   # 또는 주석처리
   ```
# 실행 화면 샘플 스냅샷
 ![image](https://github.com/user-attachments/assets/a64e5946-d532-4e02-9154-8c777382d517)

  
# 사진 결과
- 256x256 해상도로 배치 사이즈 16, 데이터셋 30%만 사용, 데이터 증강 허용으로 epoch30 중에서 5까지의 결과.
- 1epoch 당 30~40 분 소요 
  ![epoch_5](https://github.com/user-attachments/assets/a76e971f-ffcd-442d-84bd-56f9dc423b0e)

#### Train 실행후 샘플 이미지 저장 해석:

- Input (입력)
   - 원본 컬러 이미지를 흑백으로 변환하고 일부 영역을 마스킹(검은색 영역)한 이미지 모델이 
   - 복원해야 할 대상
- Restored (복원)
   - 모델이 Input을 받아서 컬러로 복원하고 마스킹된 부분을 채워넣은 결과물
   - 모델이 예측/생성한 이미지
- Ground Truth (정답)
   - 원본 컬러 이미지
   - 모델이 최대한 이 이미지와 비슷하게 복원하는 것이 목표
   - 생성된 것이 아닌 실제 원본 이미지
