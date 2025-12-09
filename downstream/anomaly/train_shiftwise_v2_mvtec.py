# ===============================================================
# train_U_mvtec_seg_full_stable.py  (Only SW_v2 Version)
# ===============================================================

import os, argparse, random, json, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from copy import deepcopy
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import sys
import os

cudnn.benchmark = False
cudnn.deterministic = True
torch.set_float32_matmul_precision('medium')

# ===============================================================
#  Import SW_v2 Backbone
# ===============================================================
#sys.path.append(os.path.expanduser('~/shift-wiseConv'))
from backbones.SW_v2_unirep import (
    ShiftWise_v2_tiny,
    ShiftWise_v2_tiny_mvtec                
)

# ===============================================================
# Visualization
# ===============================================================
@torch.no_grad()
def save_predictions(model, loader, device, save_dir="results_vis", thr=0.5, n_samples=5, seed=42):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    random.seed(seed)
    torch.manual_seed(seed)

    saved = 0
    for imgs, masks, img_p, _ in loader:
        if saved >= n_samples:
            break

        imgs = imgs.to(device).float()
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu()
        probs = F.interpolate(probs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        preds = (probs > thr).float()

        for b in range(imgs.shape[0]):
            if saved >= n_samples:
                break

            name = os.path.splitext(os.path.basename(img_p[b]))[0]
            img_np = (imgs[b].cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
            gt_np = (masks[b].cpu().squeeze().numpy() * 255).astype(np.uint8)
            pred_np = (preds[b].cpu().squeeze().numpy() * 255).astype(np.uint8)

            overlay = img_np.copy()
            overlay[pred_np > 128] = [255, 0, 0]

            fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            axes[0].imshow(img_np); axes[0].set_title("Input")
            axes[1].imshow(gt_np, cmap="gray"); axes[1].set_title("GT")
            axes[2].imshow(pred_np, cmap="gray"); axes[2].set_title(f"Pred>{thr}")
            axes[3].imshow(overlay); axes[3].set_title("Overlay")
            for ax in axes: ax.axis("off")

            plt.tight_layout()
            plt.savefig(f"{save_dir}/{saved:02d}_{name}.png", dpi=300)
            plt.close()

            saved += 1

    print(f"✅ Saved {saved} predictions to {save_dir}/")

# ===============================================================
# Dataset (MVTec AD)
# ===============================================================
class MVTecSegDataset(Dataset):
    def __init__(self, root, category="bottle", phase="train",
                 img_size=224, aug=False, train_split=0.8, seed=42):

        self.root = root
        self.category = category
        self.phase = phase
        self.img_size = img_size
        self.aug = aug

        img_root = os.path.join(root, category)
        train_good = os.path.join(img_root, "train", "good")
        test_dir = os.path.join(img_root, "test")
        gt_root = os.path.join(img_root, "ground_truth")

        self.DEFECTS = [
            d for d in sorted(os.listdir(test_dir))
            if d != "good" and os.path.isdir(os.path.join(test_dir, d))
        ]
        print(f"📂 [{category}] Defects: {self.DEFECTS}")

        good_imgs = [
            os.path.join(train_good, f)
            for f in os.listdir(train_good)
            if f.endswith((".png", ".jpg"))
        ]

        defect_imgs = []
        for d in self.DEFECTS:
            ddir = os.path.join(test_dir, d)
            for f in os.listdir(ddir):
                if not f.endswith((".png", ".jpg")):
                    continue

                mask_p = os.path.join(gt_root, d, f"{os.path.splitext(f)[0]}_mask.png")
                if os.path.exists(mask_p):
                    defect_imgs.append((d, os.path.join(ddir, f)))

        random.seed(seed)
        random.shuffle(defect_imgs)
        cut = int(len(defect_imgs) * train_split)
        defect_train, defect_val = defect_imgs[:cut], defect_imgs[cut:]

        self.samples = []
        if phase == "train":
            sel_good = random.sample(good_imgs, min(len(good_imgs), len(defect_train)))
            for p in sel_good:
                self.samples.append(("good", p, None))
            for d, p in defect_train:
                self.samples.append((d, p, self._mask(gt_root, d, p)))

        elif phase == "val":
            for d, p in defect_val:
                self.samples.append((d, p, self._mask(gt_root, d, p)))

        else:  # test
            for d, p in defect_imgs:
                self.samples.append((d, p, self._mask(gt_root, d, p)))

        self.img_t = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.mask_t = transforms.Compose([
            transforms.Resize((img_size, img_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        self.flip = transforms.RandomHorizontalFlip(0.5)
        self.vflip = transforms.RandomVerticalFlip(0.5)

        print(f" [{phase}] total={len(self.samples)} "
              f"(good={sum(s[0]=='good' for s in self.samples)}, "
              f"defect={sum(s[0]!='good' for s in self.samples)})")

    def _mask(self, gt_root, d, p):
        base = os.path.splitext(os.path.basename(p))[0]
        return os.path.join(gt_root, d, f"{base}_mask.png")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d, img_p, mask_p = self.samples[idx]

        img = Image.open(img_p).convert("RGB")
        img = self.img_t(img)

        if mask_p and os.path.exists(mask_p):
            mask = Image.open(mask_p).convert("L")
            mask = self.mask_t(mask)
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, self.img_size, self.img_size)

        if self.phase == "train":
            if random.random() < 0.5:
                img = self.flip(img); mask = self.flip(mask)
            if random.random() < 0.5:
                img = self.vflip(img); mask = self.vflip(mask)

        return img, mask, img_p, mask_p or "no_mask"

# ===============================================================
#  Base blocks
# ===============================================================
def IN(c): return nn.InstanceNorm2d(c, affine=True)

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=k//2, bias=False),
            IN(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class VanillaUNet_Seg(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.enc1 = ConvBNReLU(3, 64)
        self.enc2 = ConvBNReLU(64, 128)
        self.enc3 = ConvBNReLU(128, 256)
        self.enc4 = ConvBNReLU(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.dec1 = ConvBNReLU(512, 256)
        self.dec2 = ConvBNReLU(256, 128)
        self.dec3 = ConvBNReLU(128, 64)
        self.out_conv = nn.Conv2d(64, num_classes, 1)
        nn.init.constant_(self.out_conv.bias, -2.0)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d1 = self.dec1(torch.cat([self._resize_like(self.up1(e4), e3), e3], 1))
        d2 = self.dec2(torch.cat([self._resize_like(self.up2(d1), e2), e2], 1))
        d3 = self.dec3(torch.cat([self._resize_like(self.up3(d2), e1), e1], 1))
        return self.out_conv(d3)

    @staticmethod
    def _resize_like(src, target):
        return F.interpolate(src, size=target.shape[-2:], mode='bilinear', align_corners=False)

# ===============================================================
#  SW_v2 활용 버전
# ===============================================================

class SWv2_FullEncoderTiny(nn.Module):
    """
    Full ShiftWise_v2_tiny backbone used as UNet encoder.
    Outputs 4 feature maps: e1(56x56), e2(28x28), e3(14x14), e4(7x7)
    """
    def __init__(self):
        super().__init__()
        target_kernel_size = [51, 49, 47, 13, 3] 
        
        base = ShiftWise_v2_tiny(
            pretrained=False, 
            kernel_size=target_kernel_size  # 여기에 리스트를 전달하면 구조가 3x3으로 생성됩니다.
        )

        # downsample layers
        self.stem = base.downsample_layers[0]   # 3→80, 112x112→56x56
        self.down1 = base.downsample_layers[1]  # 80→160, 56→28
        self.down2 = base.downsample_layers[2]  # 160→320, 28→14
        self.down3 = base.downsample_layers[3]  # 320→640, 14→7

        # full stages (all blocks)
        self.stage0 = base.stages[0]  # depth 3
        self.stage1 = base.stages[1]  # depth 3
        self.stage2 = base.stages[2]  # depth 18
        self.stage3 = base.stages[3]  # depth 3

    def forward(self, x):
        x = self.stem(x)
        e1 = self.stage0(x)

        x = self.down1(e1)
        e2 = self.stage1(x)

        x = self.down2(e2)
        e3 = self.stage2(x)

        x = self.down3(e3)
        e4 = self.stage3(x)

        return e1, e2, e3, e4
    
class ShiftWiseUNet_SWTiny(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = SWv2_FullEncoderTiny()

        self.up1 = nn.ConvTranspose2d(640, 320, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(320, 160, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(160, 80, 2, stride=2)

        self.dec1 = ConvBNReLU(640, 320)
        self.dec2 = ConvBNReLU(320, 160)
        self.dec3 = ConvBNReLU(160, 80)
        self.out_conv = nn.Conv2d(80, num_classes, 1)
        nn.init.constant_(self.out_conv.bias, -2.0)

    def forward(self, x):
        e1, e2, e3, e4 = self.encoder(x)
        d1 = self.dec1(torch.cat([self.up1(e4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d1), e2], 1))
        d3 = self.dec3(torch.cat([self.up3(d2), e1], 1))
        return self.out_conv(d3)

# ===============================================================
#  고도화된 SW_v2 활용 버전
# ===============================================================
class SWv2_FullEncoderTiny(nn.Module):
    """Standard ShiftWise_v2_tiny"""
    def __init__(self):
        super().__init__()
        base = ShiftWise_v2_tiny(pretrained=False, kernel_size=[51, 49, 47, 13, 3])
        self.stem = base.downsample_layers[0]; self.down1 = base.downsample_layers[1]
        self.down2 = base.downsample_layers[2]; self.down3 = base.downsample_layers[3]
        self.stage0 = base.stages[0]; self.stage1 = base.stages[1]
        self.stage2 = base.stages[2]; self.stage3 = base.stages[3]

    def forward(self, x):
        x = self.stem(x); e1 = self.stage0(x)
        x = self.down1(e1); e2 = self.stage1(x)
        x = self.down2(e2); e3 = self.stage2(x)
        x = self.down3(e3); e4 = self.stage3(x)
        return e1, e2, e3, e4

class SWv2_FullEncoderTiny_PP(nn.Module):
    """[NEW] Optimized ShiftWise_v2_tiny_mvtec (IoU/F1 Boost)"""
    def __init__(self):
        super().__init__()
        # 위에서 정의한 최적화 모델 함수 호출
        base = ShiftWise_v2_tiny_mvtec(pretrained=False)
        self.stem = base.downsample_layers[0]; self.down1 = base.downsample_layers[1]
        self.down2 = base.downsample_layers[2]; self.down3 = base.downsample_layers[3]
        self.stage0 = base.stages[0]; self.stage1 = base.stages[1]
        self.stage2 = base.stages[2]; self.stage3 = base.stages[3]

    def forward(self, x):
        x = self.stem(x); e1 = self.stage0(x)
        x = self.down1(e1); e2 = self.stage1(x)
        x = self.down2(e2); e3 = self.stage2(x)
        x = self.down3(e3); e4 = self.stage3(x)
        return e1, e2, e3, e4

class ShiftWiseUNet_SWTiny(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = SWv2_FullEncoderTiny()
        self.up1 = nn.ConvTranspose2d(640, 320, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(320, 160, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(160, 80, 2, stride=2)
        self.dec1 = ConvBNReLU(640, 320); self.dec2 = ConvBNReLU(320, 160)
        self.dec3 = ConvBNReLU(160, 80); self.out_conv = nn.Conv2d(80, num_classes, 1)
        nn.init.constant_(self.out_conv.bias, -2.0)

    def forward(self, x):
        e1, e2, e3, e4 = self.encoder(x)
        d1 = self.dec1(torch.cat([self.up1(e4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d1), e2], 1))
        d3 = self.dec3(torch.cat([self.up3(d2), e1], 1))
        return self.out_conv(d3)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, max(channel // reduction, 4), bias=False), nn.ReLU(inplace=True), nn.Linear(max(channel // reduction, 4), channel, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False); self.bn1 = nn.BatchNorm2d(out_ch); self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False); self.bn2 = nn.BatchNorm2d(out_ch); self.act2 = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential() if in_ch == out_ch else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
    def forward(self, x):
        return self.act2(self.conv2(self.act1(self.bn1(self.conv1(x)))) + self.shortcut(x))

class ShiftWiseUNet_SWTinyPP(nn.Module):
    """
    [Best Performance Version]
    Encoder: Optimized SW_v2 (CoordAtt + ConvFFN)
    Decoder: Residual Blocks
    """
    def __init__(self, num_classes=1):
        super().__init__()
        # [MODIFY] Use the Optimized Encoder
        self.encoder = SWv2_FullEncoderTiny_PP()

        self.se1 = SEBlock(80); self.se2 = SEBlock(160); self.se3 = SEBlock(320); self.se4 = SEBlock(640)
        self.up1 = nn.ConvTranspose2d(640, 320, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(320, 160, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(160, 80,  2, stride=2)
        self.dec1 = ResConvBlock(640, 320); self.dec2 = ResConvBlock(320, 160); self.dec3 = ResConvBlock(160, 80)
        self.final = nn.Conv2d(80, num_classes, 1)
        nn.init.constant_(self.final.bias, -2.0)

    def forward(self, x):
        e1, e2, e3, e4 = self.encoder(x)
        d4 = self.up1(self.se4(e4))
        d1 = self.dec1(torch.cat([d4, self.se3(e3)], 1))
        d1_up = self.up2(d1)
        d2 = self.dec2(torch.cat([d1_up, self.se2(e2)], 1))
        d2_up = self.up3(d2)
        d3 = self.dec3(torch.cat([d2_up, self.se1(e1)], 1))
        return self.final(d3)

# ===============================================================
#  Loss
# ===============================================================

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum((2,3))
        union = probs.sum((2,3)) + targets.sum((2,3))
        dice = 1 - (2 * inter + self.eps) / (union + self.eps)
        return dice.mean()


def make_loss(device):
    pos_w = torch.tensor([2.0], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    dice = DiceLoss()
    return lambda logit, tgt: 0.5 * bce(logit, tgt) + 0.5 * dice(logit, tgt)


# ===============================================================
# Train
# ===============================================================

def train_segmentation(model, train_loader, val_loader, device, epochs=60, lr=1e-4):
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = make_loss(device)

    best = 1e9
    best_w = deepcopy(model.state_dict())
    patience = 10
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0

        for imgs, masks, _, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear")
            loss = criterion(logits, masks)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_loss += loss.item()

        tr_loss /= len(train_loader)

        # ---- Validation ----
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for imgs, masks, _, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear")
                val_loss += criterion(logits, masks).item()

        val_loss /= len(val_loader)
        print(f"[{ep:03d}] train {tr_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best:
            best = val_loss
            best_w = deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_w)
    return model


# ===============================================================
# Evaluation
# ===============================================================
@torch.no_grad()
def evaluate_seg(model, loader, device, thr=0.5, name="val"):
    tp=fp=fn=tn = 0

    for imgs, masks, _, _ in loader:
        logits = model(imgs.to(device))
        probs = torch.sigmoid(logits).cpu()
        probs = F.interpolate(probs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        pred = (probs > thr).bool()
        gt = (masks > 0.5).bool()

        tp += (pred & gt).sum().item()
        fp += (pred & ~gt).sum().item()
        fn += (~pred & gt).sum().item()
        tn += (~pred & ~gt).sum().item()

    dice = 2*tp / (2*tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2*precision*recall / (precision+recall+1e-8)
    acc = (tp+tn) / (tp+tn+fp+fn+1e-8)

    print(f"[{name}] Dice:{dice:.4f} | IoU:{iou:.4f} | "
          f"P:{precision:.4f} | R:{recall:.4f} | F1:{f1:.4f} | Acc:{acc:.4f}")

    return dict(dice=dice, iou=iou, precision=precision,
                recall=recall, f1=f1, acc=acc)


# ===============================================================
# Main
# ===============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", type=str, default="bottle")
    ap.add_argument("--model", type=str, default="swtiny",
                    choices=["swtiny", "swtiny_pp", "vanilla"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--root", type=str, default="./mvtec_ad/mvtec")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{args.category} | Model:{args.model} | Device:{device}")

    # Dataset
    train_ds = MVTecSegDataset(args.root, args.category, "train", aug=True)
    val_ds = MVTecSegDataset(args.root, args.category, "val")
    test_ds = MVTecSegDataset(args.root, args.category, "test")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    # Model selection
    if args.model == "vanilla":
        model = VanillaUNet_Seg()
    elif args.model == "swtiny":
        model = ShiftWiseUNet_SWTiny()
    elif args.model == "swtiny_pp":
        model = ShiftWiseUNet_SWTinyPP()
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # Train
    model = train_segmentation(model, train_loader, val_loader, device,
                               epochs=args.epochs, lr=1e-4)

    # Evaluation
    val_metrics  = evaluate_seg(model, val_loader, device, 0.5, "val")
    test_metrics = evaluate_seg(model, test_loader, device, 0.5, "test")

    os.makedirs("results", exist_ok=True)
    json.dump(
        {"val": val_metrics, "test": test_metrics},
        open(f"results/{args.category}_{args.model}_metrics.json","w"),
        indent=2
    )
    torch.save(model.state_dict(), f"results/{args.category}_{args.model}.pth")

    save_predictions(model, test_loader, device,
                     save_dir=f"results/{args.category}_{args.model}_vis")

    print("Done.")

if __name__ == "__main__":
    main()
