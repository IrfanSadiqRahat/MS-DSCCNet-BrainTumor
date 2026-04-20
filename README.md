# MS-DSCCNet-BrainTumor

> **Deep Learning-Based Multi-Class Brain Tumor Classification from MRI Using a Novel MS-DSCCNet Architecture**
>
> Irfan Sadiq Rahat, et al.
>
> ***IEEE DELCON 2025*** (4th Delhi Section Conference) · Paper #234 · Accepted

[![Conference](https://img.shields.io/badge/Conference-IEEE%20DELCON%202025-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()

---

## Abstract

MS-DSCCNet is a novel multi-scale depthwise separable convolutional and channel attention network for brain tumor MRI classification across 4 classes: **Glioma, Meningioma, Pituitary, No Tumor**.

Key innovations:
- Multi-scale feature extraction via parallel convolutional branches
- Depthwise separable convolutions for parameter efficiency
- Channel attention (SE blocks) for feature recalibration
- Competitive accuracy on Kaggle Brain Tumor MRI dataset

---

## Architecture

```
Input MRI (224×224×3)
    │
    ├── Branch 1: Conv 3×3 (small receptive field)
    ├── Branch 2: Conv 5×5 (medium receptive field)
    └── Branch 3: Conv 7×7 (large receptive field)
              │
        Concatenate
              │
    Depthwise Separable Conv × 3
              │
    Channel Attention (SE Block)
              │
    GlobalAvgPool → Dense → Softmax
```

---

## Dataset

Kaggle Brain Tumor MRI Dataset (7,023 MRI images, 4 classes)
Download: [kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

## Results

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Glioma | 0.96 | 0.97 | 0.97 |
| Meningioma | 0.93 | 0.91 | 0.92 |
| No Tumor | 0.98 | 0.99 | 0.99 |
| Pituitary | 0.97 | 0.97 | 0.97 |
| **Overall** | **0.96** | **0.96** | **0.96** |

---

## Setup

```bash
git clone https://github.com/IrfanSadiqRahat/MS-DSCCNet-BrainTumor.git
cd MS-DSCCNet-BrainTumor
pip install -r requirements.txt
# Download dataset from Kaggle, place in data/brain_tumor/
python train.py --data_dir data/brain_tumor --epochs 50
```

---

## Citation

```bibtex
@inproceedings{rahat2025msdscnet,
  title={Deep Learning-Based Multi-Class Brain Tumor Classification from MRI Using a Novel MS-DSCCNet Architecture},
  author={Rahat, Irfan Sadiq and others},
  booktitle={Fourth IEEE Delhi Section Conference (DELCON 2025)},
  year={2025},
  organization={IEEE}
}
```
