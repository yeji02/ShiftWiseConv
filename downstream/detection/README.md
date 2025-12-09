# Object Detection with ShiftwiseConv on COCO

This folder contains the configurations and instructions for object detection tasks using **ShiftwiseConv** on the COCO dataset. We utilize the [MMDetection](https://github.com/open-mmlab/mmdetection) framework.

## 1. Installation & Prerequisites

Since we use MMDetection as the base framework, you need to clone the official repository and install the dependencies first.

### Step 1: Clone MMDetection (v2.28.2)
```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v2.28.2
```

### Step 2: Install Dependencies
```bash
pip install scipy
pip install terminaltables
pip install -U cython
pip install "git+https://gitee.com/pursuit_zhangyu/cocoapi.git#subdirectory=PythonAPI"
pip install numpy==1.23.0

# Install MMDetection dependencies
pip install openmim
mim install mmcv-full==1.7.1  # Adjust version according to your CUDA/PyTorch
pip install -r requirements/build.txt
pip install -v -e .
```

## 2. Integration (Patching ShiftwiseConv)

You need to register `ShiftwiseConv` into the MMDetection framework. Please follow these steps to move the necessary files from this repository to the cloned `mmdetection` folder.

### Step 1: Copy Backbone File
Copy the backbone file to MMDetection's backbone directory.
* **Note:** The only difference between `mmdetection/SW_v2_unirep.py` and the original `SW_v2_unirep.py` (for ImageNet) is the `@BACKBONES.register_module` decorator.

```bash
# Assuming you are in the root of 'ShiftWiseConv' repo
cp backbones/SW_v2_unirep.py ../mmdetection/mmdet/models/backbones/
```

### Step 2: Register Backbone
Modify `mmdetection/mmdet/models/backbones/__init__.py` to import the new backbone.

Add the following lines to `__init__.py`:
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
Move the configuration files and custom optimizers to the MMDetection directory.

```bash
# 1. Create a config folder for ShiftWiseConv
mkdir -p ../mmdetection/configs/SW/

# 2. Copy config files
cp detection/configs/*.py ../mmdetection/configs/SW/

# 3. Copy Optimizer files (if you have custom optimizers in your repo)
# cp -r detection/mmdet/core/optimizers/* ../mmdetection/mmdet/core/optimizers/
```

## 3. Data Preparation

Prepare the COCO dataset according to the guidelines in [MMDetection v2.28.1](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md).

```bash
cd ../mmdetection
mkdir data
ln -s /path/to/your/coco/dataset data/coco
```

## 4. Training

You can finetune our released pretrained weights.
* **Pretrained Weights:** The path of pretrained models corresponds to the `checkpoint_file` used in `upernet_sw_tiny_512_80k_ade20k_ss` (Refer to the main README).

### Single GPU Training
To train on a single GPU:

```bash
python tools/train.py configs/SW/cascade_mask_rcnn_sw_tiny_120_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k.py \
    --work-dir ctest \
    --seed 0 \
    --deterministic \
    --gpu-ids 0
```

### Multi-GPU Training
To train on multiple GPUs (e.g., 4 GPUs):

```bash
export CUDA_VISIBLE_DEVICES=2,4,5,6
bash tools/dist_train.sh configs/SW/cascade_mask_rcnn_sw_tiny_120_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k.py 4 \
    --work-dir ctest \
    --auto-resume \
    --seed 0 \
    --deterministic
```

## 5. Evaluation & Visualization

### Test
Evaluate the model (`bbox` and `segm` metrics):
```bash
python tools/test.py configs/SW/cascade_mask_rcnn_sw_tiny_mstrain_480-800_adamw_3x_coco_in1k.py \
    ctest77/epoch_2.pth \
    --eval bbox segm
```

### Visualization
Visualize detection results:
```bash
python tools/test.py configs/SW/in1k_fpn_3x_coco.py \
    in1k_3x_coco_ap51.75.pth \
    --show-dir your_showdir
```

### FLOPs Analysis
Calculate FLOPs and Parameters using `get_flops.py`.

```bash
python tools/analysis_tools/get_flops.py work_dirs/tiny_flops.py --shape 1280 800
```

**Results:**
```text
tiny
==============================
Input shape: (3, 1280, 800)
Flops: 751.21 GFLOPs
Params: 86.89 M
==============================
small
==============================
Input shape: (3, 1280, 800)
Flops: 839.27 GFLOPs
Params: 110.48 M
==============================
```