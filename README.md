# 이미지 색상화 및 복원 AI

흑백 이미지의 색상화 및 손실된 부분을 복원하는 AI 모델

## 프로젝트 구조
```
.
├── data/               # 데이터셋 (train, test)
├── src/               # 소스 코드
│   ├── models/        # 모델 구현
│   ├── utils/         # 유틸리티 함수
│   └── datasets/      # 데이터셋 처리
├── configs/           # 설정 파일
└── notebooks/         # 실험 노트북
```

## 설치 방법

```bash
pip install -r requirements.txt
```

## 사용 방법

1. 데이터 준비
2. 모델 학습
3. 추론 및 평가

## 주요 기능

- 흑백 이미지 색상화
- 이미지 손실 부분 복원
- 고품질 이미지 생성 