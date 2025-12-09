# Semantic Segmentation with ShiftwiseConv on ADE20K

This folder contains the configurations and instructions for semantic segmentation tasks using **ShiftwiseConv** on the ADE20K dataset. We utilize the [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) framework.

## 1. Installation & Prerequisites

We use **MMSegmentation v0.30.0**. Please follow the steps below to clone the repository and install dependencies.

### Step 1: Clone MMSegmentation
```bash
git clone [https://github.com/open-mmlab/mmsegmentation.git](https://github.com/open-mmlab/mmsegmentation.git)
cd mmsegmentation
git checkout v0.30.0
```

### Step 2: Install Dependencies
```bash
pip install prettytable
pip install -U openmim
mim install mmcv-full==1.7.2
```

### Step 3: Library Modifications (Important Fixes)
Based on your environment, you may need to manually modify some library files to prevent errors.

**1. Fix `mmcv` config verification**
Open your `mmcv/utils/config.py` (located in your python site-packages) and **remove** the `verify` related code around line 502.
```bash
# Example path (adjust to your environment)
vim /root/miniconda3/lib/python3.8/site-packages/mmcv/utils/config.py
# Action: Remove or comment out the verify logic
```

**2. Fix `numpy` type issue**
Open `numpy/core/function_base.py` and cast the `num` variable to `int` around line 120.
```bash
# Example path (adjust to your environment)
vim /root/miniconda3/lib/python3.8/site-packages/numpy/core/function_base.py
# Action: Change 'num' to 'int(num)'
```

## 2. Integration (Patching ShiftwiseConv)

You need to register `ShiftwiseConv` into the MMSegmentation framework.

### Step 1: Copy Backbone File
Copy the backbone file to MMSegmentation's backbone directory.
* **Note:** Ensure that `SW_v2_unirep.py` has the `@BACKBONES.register_module` decorator enabled (this is the only difference from the ImageNet version).

```bash
# Assuming you are in the root of 'ShiftWiseConv' repo
# If you have a specific file for segmentation:
cp segmentation/SW_v2_unirep.py ../mmsegmentation/mmseg/models/backbones/
```

### Step 2: Register Backbone
Modify `mmsegmentation/mmseg/models/backbones/__init__.py` to import the new backbone.

Add the following lines:
```python
# ... (existing imports)
from .SW_v2_unirep import ShiftWise_v2

__all__ = [
    'ResNet', 
    # ... 
    'ShiftWise_v2'
]
```

### Step 3: Copy Configs & Optimizers
Move the configuration files and optimizer scripts to the MMSegmentation directory.

```bash
# 1. Create a config folder for ShiftWiseConv
mkdir -p ../mmsegmentation/configs/SW/

# 2. Copy config files
cp segmentation/configs/*.py ../mmsegmentation/configs/SW/

# 3. Copy Optimizer files (if applicable)
# cp -r segmentation/mmseg/core/optimizers/* ../mmsegmentation/mmseg/core/optimizers/
```

## 3. Data Preparation

Prepare the ADE20K dataset and link it to the `data` folder.

```bash
cd ../mmsegmentation
mkdir data
ln -s /path/to/your/ade20k_2016/ data/ade
```

## 4. Training

You can finetune our released pretrained weights.
* **Pretrained Weights:** The path of pretrained models corresponds to the `checkpoint_file` defined in the config file (e.g., `upernet_sw_tiny_512_80k_ade20k_ss`).

### Multi-GPU Training
To train on 4 GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash tools/dist_train.sh configs/SW/upernet_sw_tiny_512_160k_ade20k_ss.py 4 \
    --work-dir ADE20_SW_51_sparse_1000ite/ \
    --auto-resume \
    --seed 0 \
    --deterministic
```

## 5. Evaluation

### Test
To evaluate the trained model (`mIoU` metric):

```bash
# Single GPU Test
python tools/test.py workdirs/t77ms/upernet_sw_tiny_512_160k_ade20k_may_ms.py \
    workdirs/t77ms/latest.pth \
    --eval mIoU

# Multi-GPU Test (e.g., 2 GPUs)
CUDA_VISIBLE_DEVICES=4,5 bash tools/dist_test.sh workdirs/s06_t77best/ss_test.py \
    workdirs/s06_t77best/latest.pth 2 \
    --eval mIoU
```

### FLOPs Analysis
Calculate FLOPs and Parameters using `get_flops.py`.

```bash
python tools/get_flops.py workdirs/t77ms/upernet_sw_tiny_512_160k_ade20k_may_ms.py --shape 512 2048
```

**Results:**
```text
tiny
    rep1 path4
    ==============================
    Input shape: (3, 512, 2048)
    Flops: 948.12 GFLOPs
    Params: 61.73 M
    ==============================

    rep2 path4
    ==============================
    Input shape: (3, 512, 2048)
    Flops: 953.81 GFLOPs
    Params: 62.51 M
    ==============================
small:rep1 path4
    ==============================
    Input shape: (3, 512, 2048)
    Flops: 1039.21 GFLOPs
    Params: 87.5 M
    ==============================
```