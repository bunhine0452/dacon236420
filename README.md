
# 이미지 색상화 및 복원 AI

흑백 이미지의 색상화 및 손실된 부분을 복원하는 AI 모델. (sample 데이터 셋 사용)
## Unet 모델 설명
- https://wikidocs.net/148870
## 프로젝트 구조
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
- https://dacon.io/competitions/official/236420/overview/description
- data 폴더 생성 후 폴더 내에 데이터 셋 파일 복사
- git clone 후 폴더 내에 데이터 셋 파일 복사
```bash
git clone https://github.com/bunhine0452/dacon236420
```


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
```

2. 데이터 준비
- `data` 디렉토리에 데이터셋 구성
- CSV 파일 형식 확인 (input_image_path, gt_image_path 컬럼 필요)

## 사용 방법

### 1. 학습 실행
```bash
python src/train.py
```
- 실행 시 다음 선택 사항 제공:
  - 데이터 증강 사용 여부 (y/n)
  - 학습 설정 확인 및 시작 여부 (y/n)

### 2. 추론 실행
```bash
python src/inference.py
```
- `checkpoints/best_model.pth` 사용
- 결과는 `predictions` 디렉토리에 저장

## 주요 파라미터 수정

### 1. 이미지 크기 변경
```python
# src/datasets/dataset.py
ImageDataset(..., img_size=128)  # 128x128 크기로 변경
```

### 2. 배치 크기 조정
```python
# src/train.py
batch_size = 32  # 배치 크기 변경
```

### 3. 학습률 설정
```python
# src/train.py
learning_rate = 0.001  # 학습률 변경
```

### 4. 데이터 증강 수정
```python
# src/datasets/dataset.py
self.transform = Compose([
    Resize(img_size, img_size),
    # 원하는 증강 기법 추가/제거
])
```

## 성능 최적화

1. 학습 속도 향상:
   - 이미지 크기 축소 (128x128 권장)
   - 배치 크기 증가 (32 이상)
   - 데이터 증강 최소화
   - Mixed Precision Training 사용

2. 메모리 사용량 감소:
   - 작은 이미지 크기 사용
   - 적절한 배치 크기 선택
   - 필요 없는 데이터 증강 제거

3. 학습 품질 향상:
   - 데이터 증강 활성화
   - 적절한 에폭 수 설정 (30-50)
   - 손실 함수 가중치 조정

## 문제 해결

1. GPU 메모리 부족:
   - 배치 크기 감소
   - 이미지 크기 축소
   - 모델 크기 축소

2. 학습 불안정:
   - 학습률 감소
   - 배치 크기 조정
   - 손실 함수 가중치 조정

3. 낮은 성능:
   - 데이터 증강 활성화
   - 에폭 수 증가
   - 모델 구조 수정