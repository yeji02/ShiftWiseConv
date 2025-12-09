# Anomaly Detection with ShiftwiseConv on MVTec AD

This folder contains the training and evaluation scripts for Anomaly Detection tasks on the **MVTec AD** dataset using **ShiftwiseConv** based U-Net architectures.

We provide three model variants, including a vanilla U-Net and two ShiftwiseConv-based models optimized for anomaly segmentation.

## 1. Prerequisites

Make sure you have installed the main requirements in the root directory.
Additional requirements for visualization:

```bash
pip install matplotlib pillow
```

## 2. Dataset Preparation

Please download the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract it.
The directory structure should look like this:

```text
mvtec_ad/
└── mvtec/
    ├── bottle/
    │   ├── train/
    │   │   └── good/
    │   ├── test/
    │   │   ├── good/
    │   │   ├── broken_large/
    │   │   └── ...
    │   └── ground_truth/
    │       ├── broken_large/
    │       └── ...
    ├── cable/
    └── ...
```

> **Note:** You can symlink your dataset to `./mvtec_ad/mvtec` or specify the path using the `--root` argument.

## 3. Models

This script supports three types of architectures via the `--model` argument:

| Model Name | Description | Key Features |
| :--- | :--- | :--- |
| **`swtiny`** | **ShiftWiseUNet** | Encoder: SW_v2_tiny<br>Decoder: Standard U-Net Decoder |
| **`swtiny_pp`** | **ShiftWiseUNet++** (Recommended) | Encoder: **Optimized SW_v2** (CoordAtt + ConvFFN)<br>Decoder: **Residual Blocks** + **SE Blocks** |
| **`vanilla`** | Baseline U-Net | Standard CNN-based U-Net (ResNet-like blocks) |

## 4. Usage

### Training & Evaluation

To train the model on a specific category (e.g., `bottle`), run the following command:

```bash
# Basic usage (SW-Tiny model)
python configs/train_U_mvtec_seg_full_stable.py --category bottle --model swtiny
```

### Best Performance Setting (`swtiny_pp`)

For the best performance, use the `swtiny_pp` model which includes attention mechanisms and an improved decoder.

```bash
python configs/train_U_mvtec_seg_full_stable.py \
    --category transistor \
    --model swtiny_pp \
    --epochs 100 \
    --batch 4
```

### Arguments

* `--category`: The category of MVTec AD to train on (e.g., `bottle`, `hazelnut`, `transistor`). Default: `bottle`.
* `--model`: Model architecture (`swtiny`, `swtiny_pp`, `vanilla`). Default: `swtiny`.
* `--root`: Path to the MVTec AD dataset root. Default: `./mvtec_ad/mvtec`.
* `--epochs`: Number of training epochs. Default: `60`.
* `--batch`: Batch size. Default: `4`.

## 5. Results & Visualization

After running the script, a `results/` folder will be created containing:

* **`{category}_{model}.pth`**: Saved model weights.
* **`{category}_{model}_metrics.json`**: Evaluation metrics (IoU, F1, Dice, etc.) for validation and test sets.
* **`{category}_{model}_vis/`**: Visualization of segmentation results (Input, GT, Prediction, Overlay).

**Visualization Example:**
The script automatically saves prediction results like the following:

| Input | Ground Truth | Prediction | Overlay |
| :---: | :---: | :---: | :---: |
| (Original Image) | (Defect Mask) | (Predicted Mask) | (Red Overlay) |