# Multimodal Bottleneck Transformer (MBT) in PyTorch

This project is a PyTorch implementation of the architecture proposed in the paper **"Attention Bottlenecks for Multimodal Fusion (NeurIPS 2021)"**, which efficiently fuses video and audio data. It includes the training results of an Emotion Recognition model using the RAVDESS multimodal dataset.

## Project Highlights
* **Mid-Fusion Architecture**: Designed to pass through independent early layers before exchanging only essential information via Bottleneck tokens at intermediate stages, rather than mixing visual and auditory information from the beginning.
* **High-Speed Data Pipeline**: Implemented a high-speed DataLoader that caches pre-processed `.pt` tensors (completing MP4 video decoding and spectrogram conversion beforehand) to resolve GPU (A100) performance bottlenecks.
* **Performance**: Achieved a training accuracy of **99.18%** in just 10 epochs.

## Training Results
Training was conducted on an A100 80GB GPU environment applying the high-speed DataLoader. (Taking approx. 37 seconds per epoch)

| Epoch | Loss | Accuracy (%) |
| :---: | :---: | :---: |
| 1 | 0.5081 | 84.91 |
| 2 | 0.1584 | 95.47 |
| 3 | 0.1112 | 96.49 |
| 4 | 0.0958 | 97.21 |
| 5 | 0.0512 | 98.41 |
| 6 | 0.0627 | 98.00 |
| 7 | 0.0622 | 98.12 |
| 8 | 0.0296 | 99.27 |
| 9 | 0.0240 | 99.35 |
| **10** | **0.0260** | **99.18** |

## Inference Visualization
The results of the model predicting the speaker's emotion by synthesizing visual (video frame) and auditory (audio spectrogram) information.

<img width="1718" height="2490" alt="test_image" src="https://github.com/user-attachments/assets/91b2e011-1492-407f-9275-4599972883e8" />

* **Left (Vision)**: The middle frame image of the input video analyzed by the model.
* **Center (Audio)**: A visualization of the input audio waveform converted into a Mel-Spectrogram.
* **Right (Prediction)**: The probability distribution across 8 emotion classes finally predicted by the MBT model through bottleneck attention. (Green: Ground truth)

## Requirements & How to Run

### 1. Requirements
```bash
pip install torch torchvision torchaudio librosa opencv-python matplotlib tqdm
```

### 2. Dataset Preparation
Download the RAVDESS Audio-Visual dataset, then run the data preprocessing script in this repository to convert the data into high-speed loading .pt tensor files.

### 3. Model Training & Test
Execute the cells in the Jupyter Notebook (*.ipynb) file in order to train the model and visualize the inference results.

References
Paper: [Attention Bottlenecks for Multimodal Fusion (Nagrani et al., 2021)](https://arxiv.org/abs/2107.00135)

Dataset: [RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)](https://www.kaggle.com/datasets/orvile/ravdess-dataset)

================================================================
# 한글 버전

# Multimodal Bottleneck Transformer (MBT) in PyTorch

본 프로젝트는 비디오와 오디오 데이터를 효율적으로 융합하는 논문 "Attention Bottlenecks for Multimodal Fusion (NeurIPS 2021)"의 아키텍처를 PyTorch로 구현하고, RAVDESS 멀티모달 데이터셋을 활용해 감정 인식(Emotion Recognition) 모델을 학습시킨 결과물입니다.

## 📌 Project Highlights
* **Mid-Fusion Architecture**: 시각과 청각 정보를 처음부터 섞지 않고, 독립적인 초기 레이어를 거친 후 중간 단계부터 병목(Bottleneck) 토큰을 통해 핵심 정보만 교환하도록 설계했습니다.
* **High-Speed Data Pipeline**: GPU(A100)의 성능 병목 현상을 해결하기 위해, MP4 비디오 디코딩과 스펙트로그램 변환을 사전 완료하여 `.pt` 텐서로 캐싱하는 고속 DataLoader를 구현했습니다.
* **Performance**: 학습 10 Epoch 만에 99.18%의 훈련 정확도를 달성했습니다.

## 📊 Training Results
A100 80GB GPU 환경에서 고속 데이터로더를 적용하여 학습을 진행했습니다. (1 Epoch 당 약 37초 소요)

| Epoch | Loss | Accuracy (%) |
| :---: | :---: | :---: |
| 1 | 0.5081 | 84.91 |
| 2 | 0.1584 | 95.47 |
| 3 | 0.1112 | 96.49 |
| 4 | 0.0958 | 97.21 |
| 5 | 0.0512 | 98.41 |
| 6 | 0.0627 | 98.00 |
| 7 | 0.0622 | 98.12 |
| 8 | 0.0296 | 99.27 |
| 9 | 0.0240 | 99.35 |
| **10** | **0.0260** | **99.18** |

## 👀 Inference Visualization
모델이 시각(비디오 프레임)과 청각(오디오 스펙트로그램) 정보를 종합하여 화자의 감정을 예측한 결과입니다.
<img width="1718" height="2490" alt="test_image" src="https://github.com/user-attachments/assets/025dc5ce-afdd-47c5-b51c-d06c46110b85" />

* **Left (Vision)**: 모델이 분석한 입력 비디오의 중간 프레임 이미지입니다.
* **Center (Audio)**: 입력 오디오 파형을 멜 스펙트로그램(Mel-Spectrogram)으로 변환한 시각화 자료입니다.
* **Right (Prediction)**: MBT 모델이 병목 어텐션을 거쳐 최종적으로 예측한 8가지 감정 클래스에 대한 확률 분포입니다. (초록색: 실제 정답)

## 🛠️ Requirements & How to Run

### 1. Requirements
```bash
pip install torch torchvision torchaudio librosa opencv-python matplotlib tqdm
```
### 2. Dataset Preparation
RAVDESS Audio-Visual 데이터셋을 다운로드한 후, 본 레포지토리의 데이터 전처리 스크립트를 실행하여 .pt 형태의 고속 로딩 텐서 파일로 변환합니다.

### 3. Model Training & Test
주피터 노트북(*.ipynb) 파일의 셀을 순서대로 실행하여 모델을 학습하고 추론 결과를 시각화할 수 있습니다.
