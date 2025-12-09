# Downstream Tasks: Segmentation, Detection, and Anomaly Detection

This repository supports various downstream tasks including semantic segmentation on ADE20K, object detection/segmentation on MS COCO, and anomaly detection on MVTec AD.

## Environment Setup
The conda environment is generally compatible across detection and segmentation tasks. However, please refer to the specific README in each folder for detailed requirements.

---

## 1. Object Detection (COCO)
- **Framework:** mmdetection v2.28.2
- **Dataset:** MS COCO 2017
- **Instructions:** A series of files must be relocated and configured. Please follow the detailed [**Detection Instructions**](detection/README.md).

## 2. Semantic Segmentation (ADE20K)
- **Framework:** mmsegmentation v0.30.0
- **Dataset:** ADE20K 2016
- **Instructions:** Files must be patched into the framework. Please follow the detailed [**Segmentation Instructions**](segmentation/README.md).

> **Troubleshooting:**
> If you encounter the `semaphore_tracker` warning during training/testing:
> ```bash
> multiprocessing/semaphore_tracker.py:144: UserWarning: semaphore_tracker: There appear to be 4 leaked semaphores to clean up at shutdown len(cache))
> ```
> You can suppress it by running:
> ```bash
> export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
> ```
> **Note:** The `img_ratios` parameter in `MultiScaleFlipAug` acts as the switch to enable or disable multi-scale testing.

## 3. Anomaly Detection (MVTec AD)
- **Framework:** Pure PyTorch (No MM-framework required)
- **Dataset:** MVTec AD
- **Task:** Anomaly Segmentation & Classification
- **Instructions:** We provide a custom U-Net based training script. Please refer to the [**Anomaly Detection Instructions**](anomaly/README.md).

---