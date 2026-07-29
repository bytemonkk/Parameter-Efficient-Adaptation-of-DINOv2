"""
DINOv2 Backbone — Multi-Decoder Comparison
===========================================
Decoders evaluated:
  1. CNN          — Simple Conv stack (your baseline)
  2. FPN          — Feature Pyramid Network-style decoder
  3. UPerNet      — Pooling pyramid + lateral connections
  4. MLP (SegFmt) — Segformer-style lightweight MLP head

Each decoder is trained independently with the same frozen DINOv2 backbone.
A final summary table is printed after all runs.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
class CFG:
    DATA_ROOT   = r"C:\Users\Dell\Downloads\archive"
    IMAGE_SIZE  = 560
    BATCH_SIZE  = 4
    NUM_WORKERS = 0
    EPOCHS      = 30
    LR          = 1e-4
    NUM_CLASSES = 8          # 0=background, 1-7 semantic classes
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG()

CLASS_NAMES = [
    "Background",   # 0  — ignored in metrics
    "Building",     # 1
    "Road",         # 2
    "Water",        # 3
    "Barren",       # 4
    "Forest",       # 5
    "Agriculture",  # 6
]
# NOTE: LoveDA has 7 classes (0-6); adjust NUM_CLASSES / CLASS_NAMES if your
#       dataset labels differ.

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class LoveDADataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
        mask  = cv2.resize(
            mask, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        mask  = torch.tensor(mask, dtype=torch.long)
        return image, mask


def collect_split(split_root):
    images, masks = [], []
    for domain in ["Rural", "Urban"]:
        img_dir  = os.path.join(split_root, domain, "images_png")
        mask_dir = os.path.join(split_root, domain, "masks_png")
        for f in sorted(os.listdir(img_dir)):
            images.append(os.path.join(img_dir,  f))
            masks.append( os.path.join(mask_dir, f))
    return images, masks


train_imgs, train_masks = collect_split(os.path.join(cfg.DATA_ROOT, "Train", "Train"))
val_imgs,   val_masks   = collect_split(os.path.join(cfg.DATA_ROOT, "Val",   "Val"))

train_loader = DataLoader(
    LoveDADataset(train_imgs, train_masks),
    batch_size=cfg.BATCH_SIZE, shuffle=True,
    num_workers=cfg.NUM_WORKERS, pin_memory=True,
)
val_loader = DataLoader(
    LoveDADataset(val_imgs, val_masks),
    batch_size=cfg.BATCH_SIZE, shuffle=False,
    num_workers=cfg.NUM_WORKERS, pin_memory=True,
)

# ─────────────────────────────────────────────
# SHARED FROZEN DINOv2 ENCODER  (loaded once)
# ─────────────────────────────────────────────
print("\nLoading DINOv2 encoder …")
_dino_encoder = AutoModel.from_pretrained("facebook/dinov2-base")
for p in _dino_encoder.parameters():
    p.requires_grad = False
_dino_encoder = _dino_encoder.to(cfg.DEVICE)
print("DINOv2 encoder loaded and frozen.\n")

EMBED_DIM = 768   # dinov2-base hidden size
PATCH     = 40    # IMAGE_SIZE / patch_size(14) ≈ 40


# ─────────────────────────────────────────────
# HELPER — extract patch grid from DINOv2
# ─────────────────────────────────────────────
def encode(images):
    """Returns (B, 768, H_patch, W_patch) feature map."""
    with torch.no_grad():
        out = _dino_encoder(pixel_values=images)
    tokens = out.last_hidden_state[:, 1:, :]    # drop [CLS]
    B, N, C = tokens.shape
    H = W = int(N ** 0.5)
    return tokens.permute(0, 2, 1).reshape(B, C, H, W)


# ─────────────────────────────────────────────
# DECODER DEFINITIONS
# ─────────────────────────────────────────────

# ── 1. CNN Decoder (your baseline) ──────────
class CNNDecoder(nn.Module):
    """Simple 3-layer Conv stack + bilinear upsample."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(EMBED_DIM, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512,       256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256,       128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
        )
        self.head = nn.Conv2d(128, cfg.NUM_CLASSES, 1)

    def forward(self, feat, img_size):
        x = self.net(feat)
        x = F.interpolate(x, size=img_size, mode="bilinear", align_corners=False)
        return self.head(x)


# ── 2. FPN Decoder ───────────────────────────
class FPNDecoder(nn.Module):
    """
    Feature Pyramid Network-style decoder.
    We create 4 pseudo-scale levels by progressively downsampling
    the patch feature map, then merge top-down with lateral connections.
    """
    def __init__(self, fpn_channels=256):
        super().__init__()
        self.C = fpn_channels

        # lateral projections (same spatial size, different receptive fields)
        self.lat4 = nn.Conv2d(EMBED_DIM, self.C, 1)   # 1× (patch grid)
        self.lat3 = nn.Conv2d(EMBED_DIM, self.C, 1)   # 2× (avg-pooled)
        self.lat2 = nn.Conv2d(EMBED_DIM, self.C, 1)   # 4× (avg-pooled)
        self.lat1 = nn.Conv2d(EMBED_DIM, self.C, 1)   # 8× (avg-pooled)

        # top-down smoothing convs
        self.smooth4 = nn.Sequential(nn.Conv2d(self.C, self.C, 3, padding=1), nn.ReLU(True))
        self.smooth3 = nn.Sequential(nn.Conv2d(self.C, self.C, 3, padding=1), nn.ReLU(True))
        self.smooth2 = nn.Sequential(nn.Conv2d(self.C, self.C, 3, padding=1), nn.ReLU(True))
        self.smooth1 = nn.Sequential(nn.Conv2d(self.C, self.C, 3, padding=1), nn.ReLU(True))

        # merge all levels → head
        self.merge = nn.Sequential(
            nn.Conv2d(self.C * 4, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 128, 3, padding=1),         nn.BatchNorm2d(128), nn.ReLU(True),
        )
        self.head = nn.Conv2d(128, cfg.NUM_CLASSES, 1)

    def forward(self, feat, img_size):
        # build 4 pseudo-scales by pooling
        f4 = feat                                           # (B, 768, H, W)
        f3 = F.avg_pool2d(feat, 2, 2)
        f2 = F.avg_pool2d(feat, 4, 4)
        f1 = F.avg_pool2d(feat, 8, 8)

        # lateral
        p4 = self.lat4(f4)
        p3 = self.lat3(f3)
        p2 = self.lat2(f2)
        p1 = self.lat1(f1)

        # top-down merge
        p3 = self.smooth3(p3 + F.interpolate(p4, size=p3.shape[-2:], mode="nearest"))
        p2 = self.smooth2(p2 + F.interpolate(p3, size=p2.shape[-2:], mode="nearest"))
        p1 = self.smooth1(p1 + F.interpolate(p2, size=p1.shape[-2:], mode="nearest"))
        p4 = self.smooth4(p4)

        # upsample all to p4 resolution and concat
        target = p4.shape[-2:]
        fused  = torch.cat([
            p4,
            F.interpolate(p3, target, mode="bilinear", align_corners=False),
            F.interpolate(p2, target, mode="bilinear", align_corners=False),
            F.interpolate(p1, target, mode="bilinear", align_corners=False),
        ], dim=1)

        x = self.merge(fused)
        x = F.interpolate(x, size=img_size, mode="bilinear", align_corners=False)
        return self.head(x)


# ── 3. UPerNet Decoder ───────────────────────
class PPM(nn.Module):
    """Pooling Pyramid Module used in UPerNet."""
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        out = in_channels // len(pool_sizes)
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out, 1, bias=False),
                nn.BatchNorm2d(out), nn.ReLU(True),
            )
            for ps in pool_sizes
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + out * len(pool_sizes), in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

    def forward(self, x):
        h, w   = x.shape[-2:]
        parts  = [x] + [
            F.interpolate(s(x), (h, w), mode="bilinear", align_corners=False)
            for s in self.stages
        ]
        return self.bottleneck(torch.cat(parts, dim=1))


class UPerNetDecoder(nn.Module):
    """Simplified UPerNet: PPM on deepest feature + FPN-like lateral merging."""
    def __init__(self, fpn_channels=256):
        super().__init__()
        self.C    = fpn_channels
        self.ppm  = PPM(EMBED_DIM)

        # laterals for 4 pseudo-scales
        self.lat  = nn.ModuleList([nn.Conv2d(EMBED_DIM, self.C, 1) for _ in range(4)])
        self.ppm_proj = nn.Conv2d(EMBED_DIM, self.C, 1)

        self.fpn_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(self.C, self.C, 3, padding=1), nn.ReLU(True))
            for _ in range(4)
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(self.C * 4, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 128, 3, padding=1),         nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Dropout2d(0.1),
        )
        self.head = nn.Conv2d(128, cfg.NUM_CLASSES, 1)

    def forward(self, feat, img_size):
        # 4 pseudo-scales
        scales = [
            feat,
            F.avg_pool2d(feat, 2, 2),
            F.avg_pool2d(feat, 4, 4),
            F.avg_pool2d(feat, 8, 8),
        ]

        # apply PPM on coarsest scale, project
        p_ppm = self.ppm_proj(self.ppm(scales[-1]))

        # build FPN top-down
        laterals = [self.lat[i](s) for i, s in enumerate(scales)]
        laterals[-1] = laterals[-1] + p_ppm   # inject PPM into coarsest

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="bilinear", align_corners=False
            )

        outs   = [self.fpn_convs[i](laterals[i]) for i in range(4)]
        target = outs[0].shape[-2:]
        fused  = torch.cat([
            F.interpolate(o, target, mode="bilinear", align_corners=False) for o in outs
        ], dim=1)

        x = self.fuse(fused)
        x = F.interpolate(x, size=img_size, mode="bilinear", align_corners=False)
        return self.head(x)


# ── 4. MLP Decoder (Segformer-style) ─────────
class MLPDecoder(nn.Module):
    """
    SegFormer-style all-MLP head.
    Each scale is linearly projected to the same channel dimension,
    then concatenated and fused with a single Conv.
    Very parameter-efficient.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.E = embed_dim

        # linear projections (implemented as 1×1 conv) for 4 scales
        self.projs = nn.ModuleList([nn.Conv2d(EMBED_DIM, self.E, 1) for _ in range(4)])

        self.fuse = nn.Sequential(
            nn.Conv2d(self.E * 4, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 128,   3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Dropout2d(0.1),
        )
        self.head = nn.Conv2d(128, cfg.NUM_CLASSES, 1)

    def forward(self, feat, img_size):
        scales = [
            feat,
            F.avg_pool2d(feat, 2, 2),
            F.avg_pool2d(feat, 4, 4),
            F.avg_pool2d(feat, 8, 8),
        ]
        target  = scales[0].shape[-2:]
        proj    = [
            F.interpolate(self.projs[i](s), target, mode="bilinear", align_corners=False)
            for i, s in enumerate(scales)
        ]
        x = self.fuse(torch.cat(proj, dim=1))
        x = F.interpolate(x, size=img_size, mode="bilinear", align_corners=False)
        return self.head(x)


# ─────────────────────────────────────────────
# WRAPPER MODEL
# ─────────────────────────────────────────────
class DINOv2Seg(nn.Module):
    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, x):
        feat = encode(x)          # (B, 768, H_p, W_p) — no grad
        return self.decoder(feat, img_size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))


# ─────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, num_classes=cfg.NUM_CLASSES, ignore_index=0, smooth=1e-6):
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.smooth       = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        dice, count = 0.0, 0
        for c in range(self.num_classes):
            if c == self.ignore_index:
                continue
            pred = probs[:, c]
            gt   = (targets == c).float()
            intersection = (pred * gt).sum()
            union        = pred.sum() + gt.sum()
            dice  += 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
            count += 1
        return dice / count


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(ignore_index=0)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return 0.5 * self.ce(logits, targets) + 0.5 * self.dice(logits, targets)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_iou(cm):
    ious = []
    for i in range(cm.shape[0]):
        tp    = cm[i, i]
        fp    = cm[:, i].sum() - tp
        fn    = cm[i, :].sum() - tp
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else 0.0)
    return float(np.mean(ious)), ious


def compute_dice_from_cm(cm):
    dices = []
    for i in range(cm.shape[0]):
        tp    = cm[i, i]
        fp    = cm[:, i].sum() - tp
        fn    = cm[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        dices.append((2 * tp) / denom if denom > 0 else 0.0)
    return float(np.mean(dices)), dices


@torch.no_grad()
def validate(model, loader):
    model.eval()
    all_preds, all_targets = [], []

    for images, masks in loader:
        images  = images.to(cfg.DEVICE)
        outputs = model(images)
        preds   = torch.argmax(outputs, dim=1).cpu().numpy()
        masks   = masks.numpy()

        valid = masks != 0
        all_preds.extend(  preds[valid].flatten())
        all_targets.extend(masks[valid].flatten())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)

    labels          = list(range(1, len(CLASS_NAMES)))   # 1..6
    cm              = confusion_matrix(all_targets, all_preds, labels=labels)
    accuracy        = accuracy_score(all_targets, all_preds)
    precision       = precision_score(all_targets, all_preds, average="macro",
                                      labels=labels, zero_division=0)
    recall          = recall_score(   all_targets, all_preds, average="macro",
                                      labels=labels, zero_division=0)
    f1              = f1_score(       all_targets, all_preds, average="macro",
                                      labels=labels, zero_division=0)
    miou, class_iou   = compute_iou(cm)
    mdice, class_dice = compute_dice_from_cm(cm)

    return {
        "accuracy":   accuracy,
        "precision":  precision,
        "recall":     recall,
        "f1":         f1,
        "miou":       miou,
        "mdice":      mdice,
        "class_iou":  class_iou,
        "class_dice": class_dice,
        "cm":         cm,
    }


# ─────────────────────────────────────────────
# TRAINING LOOP (for one decoder)
# ─────────────────────────────────────────────
def train_decoder(name, decoder):
    print(f"\n{'#'*60}")
    print(f"  TRAINING DECODER : {name}")
    print(f"{'#'*60}")

    model     = DINOv2Seg(decoder).to(cfg.DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params : {trainable:,}\n")

    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS
    )

    best_miou    = 0.0
    best_metrics = None
    t0           = time.time()

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{name}] Epoch {epoch}/{cfg.EPOCHS}")

        for images, masks in pbar:
            images = images.to(cfg.DEVICE)
            masks  = masks.to(cfg.DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        metrics  = validate(model, val_loader)
        scheduler.step()

        print(
            f"  Epoch {epoch:>2}/{cfg.EPOCHS}  "
            f"Loss={avg_loss:.4f}  "
            f"mIoU={metrics['miou']:.4f}  "
            f"mDice={metrics['mdice']:.4f}  "
            f"F1={metrics['f1']:.4f}"
        )

        if metrics["miou"] > best_miou:
            best_miou    = metrics["miou"]
            best_metrics = metrics
            torch.save(model.state_dict(), f"dinov2_{name.lower()}_best.pth")
            print(f"    ✓ Checkpoint saved  (mIoU={best_miou:.4f})")

    elapsed = time.time() - t0
    best_metrics["train_time_min"] = elapsed / 60
    best_metrics["trainable_params"] = trainable
    best_metrics["decoder_name"]   = name

    # per-decoder detailed report
    print(f"\n{'='*50}")
    print(f"  {name} — Best Validation Metrics")
    print(f"{'='*50}")
    print(f"  Accuracy  : {best_metrics['accuracy']:.4f}")
    print(f"  Precision : {best_metrics['precision']:.4f}")
    print(f"  Recall    : {best_metrics['recall']:.4f}")
    print(f"  F1 Score  : {best_metrics['f1']:.4f}")
    print(f"  mIoU      : {best_metrics['miou']:.4f}")
    print(f"  mDice     : {best_metrics['mdice']:.4f}")
    print(f"\n  Per-class IoU & Dice:")
    for i, cls in enumerate(CLASS_NAMES[1:]):
        iou  = best_metrics["class_iou"][i]
        dice = best_metrics["class_dice"][i]
        print(f"    {cls:<14}: IoU={iou:.4f}  Dice={dice:.4f}")
    print(f"\n  Confusion Matrix (classes 1-{len(CLASS_NAMES)-1}):")
    print(best_metrics["cm"])
    print(f"{'='*50}\n")

    return best_metrics


# ─────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────
def print_summary_table(results: list):
    """
    Prints a formatted comparison table across all decoders.
    """
    cols  = ["Decoder", "Params", "mIoU", "mDice", "F1", "Accuracy",
             "Precision", "Recall", "Time(min)"]
    w     = [14, 12, 8, 8, 8, 10, 11, 9, 10]
    sep   = "─" * (sum(w) + len(w) * 3 + 1)

    def row(vals):
        return "│ " + " │ ".join(str(v).ljust(w[i]) for i, v in enumerate(vals)) + " │"

    print(f"\n\n{'='*80}")
    print("  DECODER COMPARISON — FINAL SUMMARY")
    print(f"{'='*80}")
    print(sep)
    print(row(cols))
    print(sep)

    best_miou_val = max(r["miou"] for r in results)

    for r in results:
        marker = " ★" if r["miou"] == best_miou_val else ""
        vals = [
            r["decoder_name"] + marker,
            f"{r['trainable_params']:,}",
            f"{r['miou']:.4f}",
            f"{r['mdice']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['accuracy']:.4f}",
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['train_time_min']:.1f}",
        ]
        print(row(vals))

    print(sep)
    print("  ★ = best mIoU\n")

    # per-class IoU table
    n_cls = len(CLASS_NAMES) - 1   # exclude background
    print(f"\n  Per-class IoU")
    print(sep)
    hdr = ["Class"] + [r["decoder_name"] for r in results]
    hw  = [14] + [12] * len(results)

    def row2(vals):
        return "│ " + " │ ".join(str(v).ljust(hw[i]) for i, v in enumerate(vals)) + " │"

    print(row2(hdr))
    print(sep)
    for i, cls in enumerate(CLASS_NAMES[1:]):
        iou_vals = [f"{r['class_iou'][i]:.4f}" for r in results]
        print(row2([cls] + iou_vals))
    print(sep)

    # per-class Dice table
    print(f"\n  Per-class Dice")
    print(sep)
    print(row2(hdr))
    print(sep)
    for i, cls in enumerate(CLASS_NAMES[1:]):
        dice_vals = [f"{r['class_dice'][i]:.4f}" for r in results]
        print(row2([cls] + dice_vals))
    print(sep)
    print()


# ─────────────────────────────────────────────
# MAIN — run all decoders sequentially
# ─────────────────────────────────────────────
DECODERS = {
    "CNN":     CNNDecoder(),
    "FPN":     FPNDecoder(),
    "UPerNet": UPerNetDecoder(),
    "MLP":     MLPDecoder(),
}

all_results = []

for decoder_name, decoder_module in DECODERS.items():
    result = train_decoder(decoder_name, decoder_module)
    all_results.append(result)
    # free GPU memory before next run
    torch.cuda.empty_cache()

print_summary_table(all_results)
print("All done. Best checkpoints saved as  dinov2_<decoder>_best.pth")
