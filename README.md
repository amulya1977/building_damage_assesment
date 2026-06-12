# building_damage_assesment
# AI-Powered Building Damage Assessment for Disaster Response

A complete two-stage AI pipeline developed to assess building damage from drone orthomosaic imagery immediately following natural disasters. This system operates independently of pre-existing building maps and generates prioritized, ArcGIS-ready assessment maps to assist emergency response teams.

Developed at the **Indian Institute of Information Technology, Allahabad (IIIT-A)**.

## 📖 Overview

Getting fast, clear answers about damaged buildings after disasters helps save lives. Traditional manual review of drone imagery can take days. This project introduces a multi-stage deep learning pipeline that processes gigapixel raw drone imagery (GeoTIFFs) to localize buildings and classify their damage severity in minutes. 

Unlike earlier methods, this system completely removes map dependency, avoiding alignment issues and reliance on outdated base layers.

## ✨ Key Features

* **Map-Independent Localization:** Spots buildings directly from raw drone shots without needing pre-disaster maps.
* **Two-Stage Architecture:** Combines a hybrid CNN-Transformer for segmentation and a multi-axis vision transformer for classification.
* **Uncertainty Quantification:** Outputs predictions with reliability/confidence scores.
* **GIS Integration:** Generates standardized outputs containing GPS coordinates, bounding boxes, and damage labels ready for traditional GIS software (e.g., ArcGIS).

## 📊 Dataset

The model is trained and evaluated on the **CRASAR-U-DROIDs** dataset, comprising real-world aerial imagery from 10 federal disaster scenarios.
* **Size:** 52 orthomosaic GeoTIFFs (~219 GB).
* **Annotations:** 21,716 buildings labeled across four damage categories.
* **Split Strategy:** Geographically and temporally aware disaster-level splits to ensure rigorous out-of-distribution (OOD) evaluation.

## 🧠 System Architecture

### Stage 1: Building Localization (U-Net + MiT-B3)
Takes the raw orthomosaic as input and outputs a binary segmentation map of building footprints.
* **Backbone:** Mix Transformer B3 (MiT-B3) encoder paired with a U-Net decoder.
* **Attention:** Utilizes spatial and channel Squeeze-and-Excitation (scSE) blocks to recalibrate feature responses.
* **Loss Function:** 30% Binary Cross-Entropy Loss combined with Tversky loss to handle severe class imbalance.

### Stage 2: Damage Classification (MaxViT-Base)
Extracts building crops using bounding boxes from Stage 1 and classifies them into four categories: No Damage, Minor Damage, Major Damage, or Destroyed.
* **Backbone:** MaxViT-Base, pre-trained on ImageNet-21k, utilizing both Local Window Self-Attention and Global Grid Self-Attention to capture micro textures and macro structural context.
* **Advanced Training Techniques:** Class-Balanced Focal Loss, Layer-wise Learning Rate Decay (LLRD), Progressive Resizing, and Exponential Moving Average (EMA).

## 🚀 Performance Metrics

Evaluated on an OOD test set consisting of unseen disasters (Hurricanes Idalia & Michael, Mussett Bayou Fire, Mayfield Tornado).

| Component | Architecture | Key Metric | Result (OOD) |
| :--- | :--- | :--- | :--- |
| **Localization (Stage 1)** | U-Net + MiT-B3 | F1 / IoU | **F1 = 0.722**, IoU = 0.60 |
| **Classification (Stage 2)** | MaxViT-Base | Macro F1 / Accuracy | **Macro F1 = 0.69**, Acc = 72% |

*Note: 8-Fold Test Time Augmentation (TTA) was utilized during inference to significantly reduce prediction variance.*

## 🛠️ Prerequisites & Environment

This pipeline requires a high-performance Linux environment due to the massive size of the orthomosaic inputs and the computational intensity of the transformer models.

* **Language:** Python 3.10+ / C++ (for custom high-performance preprocessing logic)
* **Framework:** PyTorch (with Mixed Precision FP16 support)
* **Compute:** Minimum 24GB VRAM recommended. For full-batch training, a high-end data center GPU (e.g., NVIDIA RTX 6000 Ada with 48GB VRAM) is optimal.
* **Geospatial Libraries:** `rasterio`, `geopandas`, `scipy`

## 🔧 Installation & Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/amulyabolusani/disaster-damage-assessment.git](https://github.com/amulyabolusani/disaster-damage-assessment.git)
   cd disaster-damage-assessment
