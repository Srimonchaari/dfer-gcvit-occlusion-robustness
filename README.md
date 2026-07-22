![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/timm-GC--ViT_Base-7c3aed)
![CUDA](https://img.shields.io/badge/CUDA-AMP_Mixed_Precision-76b900?logo=nvidia&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-KMU--FED_1106_images-f59e0b)
![Accuracy](https://img.shields.io/badge/Baseline_Accuracy-95.82%25-22c55e)
![Robustness](https://img.shields.io/badge/Robustness_Gain-41.6pp-0ea5e9)
![License](https://img.shields.io/badge/License-Academic_Research-6b7280)

# Occlusion-Robust Driver Facial Expression Recognition

Reimplementation and extension of DFER-GCViT (Saadi et al., IEEE CVMI 2023)

- **Author:** Srimonchaari Padmanabhan Babu
- **Institution:** BTU Cottbus-Senftenberg, Faculty of Graphical Systems
- **First Supervisor:** Dr. Ibtissam Saadi
- **Second Supervisor:** Prof. Douglas W. Cunningham
- **Dataset:** KMU-FED (Near-Infrared Driver Face Images)

---

## What This Is

Saadi et al. (IEEE CVMI 2023) built a driver facial expression recognition system using the Global Context Vision Transformer (GC-ViT) and reported 98.27% accuracy on the KMU-FED dataset. The original study did not evaluate robustness under occlusion — what happens when a driver wears sunglasses or a surgical mask.

This research module reimplements the architecture described in that paper using the timm GC-ViT Base backbone with a modified classifier head, evaluates it on KMU-FED under the same conditions, and extends it with a synthetic occlusion study. Two model variants are trained and compared across three test conditions: clean, eye-occluded and mouth-occluded.

---

## Occlusion Conditions

The images below show how synthetic occlusion is applied to each of the 6 emotion classes in KMU-FED. The black rectangles simulate sunglasses (eye region) and a surgical mask (mouth region).

![Sample Conditions](visualisations/8_sample_conditions.png)

---

## Technology Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| PyTorch 2.x | Model training and inference |
| timm | Load pretrained GC-ViT Base (gcvit_base) |
| facenet-pytorch | MTCNN face detection and cropping |
| h5py | HDF5 data storage for fast image loading |
| scikit-learn | F1 score, confusion matrix |
| matplotlib | All 8 visualisation charts |
| numpy | Image manipulation and array ops |
| CUDA / AMP | Mixed precision training on GPU |

**Hardware used:** NVIDIA RTX 3070 (8GB VRAM), AMD Ryzen 9 5900 HS

---

## Architecture

```
NIR Face Image (224x224)
        |
   MTCNN Crop
        |
 GC-ViT Base Backbone  (pretrained ImageNet, via timm)
   Stage 1: Local Attention + Global Attention + FusedMBConv + MaxPool
   Stage 2: Local Attention + Global Attention + FusedMBConv + MaxPool
   Stage 3: Local Attention + Global Attention + FusedMBConv + MaxPool
   Stage 4: Local Attention + Global Attention + Global Context
        |
 Global Average Pooling  [Batch, 1024]
        |
 Modified Classifier Head (replaces original fc layer)
   Linear(1024 -> 512)
   BatchNorm1d(512)
   ReLU
   Dropout(0.5)
   Linear(512 -> 6)
        |
 6 Classes: Anger | Disgust | Fear | Happy | Sadness | Surprise
```

The classifier head replaces the original fully connected layer in one line in `models/gcvitt.py`:
```python
self.model.head.fc = new_layers
```

---

## The Extension

Two types of synthetic occlusion are applied programmatically on 224x224 face crops with no manual labelling:

```python
# Simulates sunglasses
def apply_eye_occlusion(image):
    occluded = image.copy()
    occluded[60:100, 30:195] = 0
    return occluded

# Simulates surgical mask
def apply_mouth_occlusion(image):
    occluded = image.copy()
    occluded[150:185, 40:185] = 0
    return occluded
```

**Model A** trains on clean images only. **Model B** trains with 50% of images occluded (25% eye, 25% mouth per class). Both models are tested on clean images from `baseline.h5`; occlusion is applied in memory at test time only.

---

## Where the Model Looks

The heatmap below shows occlusion sensitivity maps for Model A across all 6 emotion classes. Red regions indicate where the model relies most for its prediction. Mouth dominance in Happy, Anger and Surprise explains why mouth occlusion causes such a large accuracy drop.

![Occlusion Sensitivity Heatmap](visualisations/7_attention_heatmap.png)

*Generated using sliding window occlusion sensitivity (16x16 patch) on the baseline model (fold 1).*

---

## Results

|  | Model A (Baseline) | Model B (Occluded) |
|--|:--:|:--:|
| Clean Accuracy | 95.82% | 93.45% |
| Eye Occluded | 92.64% | 90.73% |
| Mouth Occluded | 35.45% | 74.73% |
| Eye Drop | 3.18 pp | 2.73 pp |
| Mouth Drop | **60.36 pp** | **18.73 pp** |

Occlusion-aware training reduced the mouth occlusion accuracy drop from 60.36 to 18.73 percentage points, a 41.6 pp recovery, at a cost of only 2.37 pp on clean accuracy.

Eye occlusion barely affects either model because GC-ViT global attention compensates using other facial regions. Mouth occlusion is catastrophic for the baseline because expressions like Happy, Surprise and Anger rely heavily on mouth signals.

![Accuracy Drop](visualisations/2_accuracy_drop.png)
![Per-Fold Accuracy](visualisations/4_per_fold_accuracy.png)

---

## Project Structure

```
dfer-gcvit-occlusion-robustness/
|
+-- models/
|   +-- gcvitt.py              GC-ViT Base + modified 5-layer classifier head
|
+-- extension/
|   +-- KMU_adapted.py         Dataset class with dynamic fold calculation
|   +-- occlusion.py           Eye and mouth occlusion functions
|   +-- train_baseline.py      10-fold training on clean images (Model A)
|   +-- train_occluded.py      10-fold training on occluded images (Model B)
|   +-- evaluate.py            Evaluate both models under 3 test conditions
|   +-- visualise.py           Generate all 8 charts
|
+-- KMU.py                     Original dataset class
+-- preprocess_kmu.py          MTCNN face crop pipeline to HDF5
+-- mainKMU.py                 Original training script
+-- matrixconKMU.py            Confusion matrix script
|
+-- results/
|   +-- evaluation.csv         All results: 2 models x 3 conditions x 10 folds
|   +-- baseline/fold_results.csv
|   +-- occluded/fold_results.csv
|
+-- visualisations/            8 PNG charts + architecture diagram
+-- Poster/                    Research poster (PDF + PPTX, first and final versions)
+-- requirements.txt
```

---

## How to Run

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Build HDF5 from raw KMU-FED images**
```bash
python preprocess_kmu.py
```

**Train Model A — baseline (all 10 folds)**
```bash
cd extension
python train_baseline.py
```

**Train Model B — occluded (all 10 folds)**
```bash
cd extension
python train_occluded.py
```

**Evaluate both models**
```bash
cd extension
python evaluate.py
```

**Generate visualisations**
```bash
cd extension
python visualise.py
```

> Note: HDF5 files (`KMUtada/baseline.h5`, `KMUtada/occluded.h5`) and model checkpoints (`results/**/*.pth`) are excluded from this repo as they exceed GitHub file limits. Run the scripts above to regenerate them.

---

## Training Setup

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Learning rate | 0.0001 |
| LR Schedule | CosineAnnealingLR (T_max=60, eta_min=1e-6) |
| Batch size | 64 effective (gradient accumulation 16x4) |
| Epochs | 60 |
| Folds | 10-fold stratified cross-validation |
| Precision | Mixed (AMP fp16 + fp32) |

Gradient accumulation was used because GC-ViT Base with batch size 64 exceeds 8 GB VRAM. Running 4 batches of 16 with accumulated gradients gives the same result mathematically.

---

## References

1. Saadi et al. (2023). Driver's Facial Expression Recognition using Global Context Vision Transformer. IEEE CVMI 2023. https://doi.org/10.1109/CVMI59935.2023.10464794
2. Hatamizadeh et al. (2022). Global context vision transformers. arXiv:2206.09959. https://arxiv.org/abs/2206.09959
3. Jeong and Ko (2018). Driver's facial expression recognition in real-time for safe driving. Sensors, 18(12), 4270. https://doi.org/10.3390/s18124270

---

For academic research use only. KMU-FED dataset terms apply.
