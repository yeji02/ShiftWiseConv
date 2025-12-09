# (CVPR2025) ShiftwiseConv: Small Convolutional Kernel with Large Kernel Effect

This repository is a **reproduction project** of the official ShiftwiseConv implementation.
In addition to reproducing the original experiments (ImageNet, COCO, ADE20K), **we have extended the evaluation to Anomaly Detection on the MVTec AD dataset** to verify the architecture's versatility in industrial inspection tasks.

---

**Original Paper:**
**ShiftwiseConv: Small Convolutional Kernel with Large Kernel Effect.** *Dachong Li, Li Li, Zhuangzhuang Chen, Jianqiang Li.* CVPR 2025

[![arxiv](https://shields.io/badge/paper-purple?logo=arxiv&style=for-the-badge)](https://arxiv.org/abs/2401.12736)
[![webpage](https://img.shields.io/badge/GitHub-blue?logo=github&style=for-the-badge)](https://github.com/lidc54/shift-wiseConv)


**TL;DR:** Our research finds that $3 \times 3$ convolutions can replace larger ones in CNNs, enhancing performance and echoing VGG's results. It also introduces novel parameter settings that have not been previously explored.

<p align="center">
<img src="SW.png" width="500" height="320">
</p>

**Abstract:**
Large kernels make standard convolutional neural networks (CNNs) great again over transformer architectures in various vision tasks. Nonetheless, recent studies meticulously designed around increasing kernel size have shown diminishing returns or stagnation in performance. Thus, the hidden factors of large kernel convolution that affect model performance remain unexplored. In this paper, we reveal that the key hidden factors of large kernels can be summarized as two separate components: extracting features at a certain granularity and fusing features by multiple pathways. To this end, we leverage the multi-path long-distance sparse dependency relationship to enhance feature utilization via the proposed Shiftwise (SW) convolution operator with a pure CNN architecture. In a wide range of vision tasks such as classification, segmentation, and detection, SW surpasses state-of-the-art transformers and CNN architectures, including SLaK and UniRepLKNet. More importantly, our experiments demonstrate that $3 \times 3$ convolutions can replace large convolutions in existing large kernel CNNs to achieve comparable effects, which may inspire follow-up works.

---

## Installation

The code is tested with CUDA 11.7, cuDNN 8.2.0, and PyTorch 1.10.0.

### 1. Environment Setup
Create a new conda virtual environment:
```bash
conda create -n shiftWise python=3.8 -y
conda activate shiftWise
```

Install [PyTorch](https://pytorch.org/) >= 1.10.0. For example:
```bash
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install timm tensorboardX six
```

### 2. Compile CUDA Module (ShiftAdd)
Install the custom CUDA kernel for ShiftwiseConv:
```bash
cd shiftadd
python setup.py install
cd ..
```

---

## Training (ImageNet-1K)

To train the model on ImageNet-1K with 8 GPUs:

```bash
python -m torch.distributed.run --master_port=29501 --nproc_per_node=8 main.py \
  --sparse --width_factor 1.0 -u 100 --sparsity 0.3 \
  --warmup_epochs 5 --sparse_init snip --prune_rate 0.3 --growth random --sparse_type rep_coarse \
  --epochs 300 --model ShiftWise_v2_tiny --drop_path 0.2 --batch_size 64 \
  --lr 4e-3 --update_freq 8 --model_ema false --model_ema_eval false \
  --data_path /root/autodl-tmp/imagenet --num_workers 64 \
  --kernel_size 51 49 47 13 3 --only_L --ghost_ratio 0.23 --sparse_gap 131 \
  --output_dir checkpoints/ 2>&1 | tee tiny.log
```
> **Note:** Please update `--data_path` to your actual ImageNet directory.

---

## Downstream Tasks & Experiments

We have organized the downstream tasks into separate directories. This includes the original tasks (Detection, Segmentation) and our **newly added Anomaly Detection** experiments.

### 1. Object Detection (COCO)
* **Framework:** MMDetection v2.28.2
* **Dataset:** MS COCO 2017
* **Status:** Reproduced
* 👉 **[Detailed Instructions](detection/README.md)**

### 2. Semantic Segmentation (ADE20K)
* **Framework:** MMSegmentation v0.30.0
* **Dataset:** ADE20K 2016
* **Status:** Reproduced
* 👉 **[Detailed Instructions](segmentation/README.md)**

### 3. Anomaly Detection (MVTec AD) [New!]
* **Framework:** Custom PyTorch Implementation (U-Net based)
* **Dataset:** MVTec AD
* **Status:** **Extended Experiment**
* We applied ShiftwiseConv as a backbone for Anomaly Segmentation tasks.
* 👉 **[Detailed Instructions](anomaly/README.md)**

> **Common Troubleshooting:**
> If you encounter `UserWarning: semaphore_tracker: There appear to be 4 leaked semaphores`, run:
> ```bash
> export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
> ```

---

## Results and Trained Models (ImageNet-1K)

| **Name** | **Resolution** | **Acc@1** | **Log** | **Model** |
|:---:|:---:|:---:|:---:|:---:|
| **SW-tiny** | 224x224 | 83.39 (300 epoch) | [SW-T](backbones/SW_300_unirep_tiny_gap131_M2_N4_g023_s03.log) | [Google Drive](https://drive.google.com/file/d/1U4DOZv5V9_7wJdqdicjp0tCmNIdRNJOc/view?usp=sharing) |

---
