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
