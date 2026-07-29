############################################################
# RAD-DINO × LoveDA SEGMENTATION PIPELINE
############################################################

import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from transformers import (
    AutoModel,
    AutoImageProcessor
)

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

from torch.utils.data import (
    Dataset,
    DataLoader
)

############################################################
# CONFIG
############################################################

class CFG:
    DATA_ROOT    = r"C:\Users\Dell\Downloads\archive"
    IMAGE_SIZE   = 518          # RAD-DINO native resolution
    BATCH_SIZE   = 2
    NUM_WORKERS  = 0
    EPOCHS       = 20
    LR           = 5e-5
    WEIGHT_DECAY = 1e-4
    NUM_CLASSES  = 8
    DICE_SMOOTH  = 1e-6
    CE_W         = 0.5
    DICE_W       = 0.4
    EDGE_W       = 0.1
    AUX_W        = 0.4
    AMP          = False
    CKPT_DIR     = "checkpoints_raddino"
    CLASS_NAMES  = ["Background", "Building", "Road", "Water",
                    "Barren", "Forest", "Agriculture", "Other"]
    MODEL_NAME   = "microsoft/rad-dino"
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG()
os.makedirs(cfg.CKPT_DIR, exist_ok=True)
print(f"Using device : {cfg.DEVICE}")

############################################################
# DATASET  (LoveDA  Rural + Urban)
############################################################

class LoveDADataset(Dataset):

    def __init__(self, image_paths, mask_paths, processor, augment=False):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.processor   = processor
        self.augment     = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # ── load ──────────────────────────────────────────
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # ── resize ────────────────────────────────────────
        image = cv2.resize(image, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
        mask  = cv2.resize(mask,  (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
                           interpolation=cv2.INTER_NEAREST)

        # ── augmentation ──────────────────────────────────
        if self.augment:
            if np.random.rand() > 0.5:
                image = np.fliplr(image).copy()
                mask  = np.fliplr(mask).copy()
            if np.random.rand() > 0.5:
                image = np.flipud(image).copy()
                mask  = np.flipud(mask).copy()
            k = np.random.randint(0, 4)
            image = np.rot90(image, k).copy()
            mask  = np.rot90(mask,  k).copy()

        # ── CLAHE (RAD-DINO pre-training style) ───────────
        gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2)
        gray  = clahe.apply(gray)
        image = np.stack([gray, gray, gray], axis=-1)

        # ── RAD-DINO processor  ───────────────────────────
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(image)
        pixel_values = self.processor(
            images=pil_img,
            return_tensors="pt"
        )["pixel_values"].squeeze(0)            # (3, H, W)

        mask = torch.tensor(mask.astype(np.int64), dtype=torch.long)
        mask = mask.clamp(0, cfg.NUM_CLASSES - 1)

        return pixel_values, mask


def collect_split(split_root):
    images, masks = [], []
    for domain in ["Rural", "Urban"]:
        img_dir  = os.path.join(split_root, domain, "images_png")
        mask_dir = os.path.join(split_root, domain, "masks_png")
        for f in sorted(os.listdir(img_dir)):
            images.append(os.path.join(img_dir,  f))
            masks.append( os.path.join(mask_dir, f))
    return images, masks

############################################################
# LOAD RAD-DINO
############################################################

print("\nLoading RAD-DINO …")

processor = AutoImageProcessor.from_pretrained(cfg.MODEL_NAME)
encoder   = AutoModel.from_pretrained(cfg.MODEL_NAME,
                                      output_hidden_states=True)
encoder   = encoder.to(cfg.DEVICE)

############################################################
# BUILDING BLOCKS
############################################################

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(6, 12, 18)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.dilated = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3,
                          padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
            for r in rates])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        n = 1 + len(rates) + 1
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * n, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1))

    def forward(self, x):
        H, W = x.shape[-2:]
        out  = [self.conv1(x)]
        for d in self.dilated:
            out.append(d(x))
        gp = F.interpolate(self.pool(x), size=(H, W),
                           mode="bilinear", align_corners=False)
        out.append(gp)
        return self.project(torch.cat(out, dim=1))


class FPNFuse(nn.Module):
    def __init__(self, encoder_dim=768, fpn_dim=256):
        super().__init__()
        self.lat = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dim, fpn_dim, 1, bias=False),
                nn.BatchNorm2d(fpn_dim), nn.ReLU(inplace=True))
            for _ in range(4)])
        self.fuse = nn.Sequential(
            nn.Conv2d(fpn_dim * 4, fpn_dim * 2, 1, bias=False),
            nn.BatchNorm2d(fpn_dim * 2), nn.ReLU(inplace=True))

    def forward(self, feats):
        H, W = feats[0].shape[-2:]
        lats = []
        for i, f in enumerate(feats):
            l = self.lat[i](f)
            if l.shape[-2:] != (H, W):
                l = F.interpolate(l, size=(H, W),
                                  mode="bilinear", align_corners=False)
            lats.append(l)
        return self.fuse(torch.cat(lats, dim=1))

############################################################
# MODEL  –  RAD-DINO encoder + FPN/ASPP decoder
############################################################

class RADDINOSeg(nn.Module):
    """
    RAD-DINO Vision Transformer encoder
    + FPN multi-scale neck
    + ASPP context aggregation
    + 4× upsampling decoder with auxiliary heads
    """

    def __init__(self, pretrained_encoder):
        super().__init__()

        # ── encoder ───────────────────────────────────────
        self.encoder     = pretrained_encoder
        self.patch_size  = 14       # RAD-DINO uses 14-px patches
        self.encoder_dim = 768      # ViT-Base hidden dim

        # fine-tune encoder (unfreeze all)
        for p in self.encoder.parameters():
            p.requires_grad = True

        # ── auto-detect feature dim & patch grid ──────────
        with torch.no_grad():
            dummy  = torch.randn(1, 3, cfg.IMAGE_SIZE,
                                 cfg.IMAGE_SIZE).to(cfg.DEVICE)
            out    = self.encoder(pixel_values=dummy,
                                  output_hidden_states=True)
            # hidden_states[0] = embedding layer, [1..12] = transformer blocks
            sample = out.hidden_states[1]          # (1, N_tokens, C)
            n_tok  = sample.shape[1] - 1           # minus CLS token
            Hp     = Wp = int(n_tok ** 0.5)
            print(f"Patch grid   : {Hp} × {Wp}  ({n_tok} patches)")
            print(f"Encoder dim  : {sample.shape[-1]}")

        self.Hp = Hp
        self.Wp = Wp

        # ── neck + decoder ────────────────────────────────
        self.fpn  = FPNFuse(encoder_dim=self.encoder_dim, fpn_dim=256)
        self.aspp = ASPP(in_ch=512, out_ch=256)

        self.up1  = nn.Upsample(scale_factor=2,
                                mode="bilinear", align_corners=False)
        self.dec1 = ResidualConvBlock(256, 256)
        self.up2  = nn.Upsample(scale_factor=2,
                                mode="bilinear", align_corners=False)
        self.dec2 = ResidualConvBlock(256, 128)
        self.up3  = nn.Upsample(scale_factor=2,
                                mode="bilinear", align_corners=False)
        self.dec3 = ResidualConvBlock(128,  64)
        self.up4  = nn.Upsample(scale_factor=2,
                                mode="bilinear", align_corners=False)
        self.dec4 = ResidualConvBlock( 64,  32)

        self.seg_head = nn.Conv2d(32,  cfg.NUM_CLASSES, kernel_size=1)
        self.aux1     = nn.Conv2d(256, cfg.NUM_CLASSES, kernel_size=1)
        self.aux2     = nn.Conv2d(128, cfg.NUM_CLASSES, kernel_size=1)
        self.aux3     = nn.Conv2d(64,  cfg.NUM_CLASSES, kernel_size=1)

    # ── helper: reshape token sequence → spatial feature map ──
    def _to_map(self, hidden):
        """
        hidden : (B, 1+N_patches, C)   — with CLS token
        returns: (B, C, Hp, Wp)
        """
        x = hidden[:, 1:, :]                           # drop CLS
        B, N, C = x.shape
        return x.permute(0, 2, 1).reshape(B, C, self.Hp, self.Wp)

    def forward(self, x):
        B, _, H, W = x.shape

        enc    = self.encoder(pixel_values=x,
                              output_hidden_states=True)
        hs     = enc.hidden_states   # tuple len 13 (embed + 12 blocks)

        # sample 4 evenly-spaced layers for FPN
        f3  = self._to_map(hs[3])
        f6  = self._to_map(hs[6])
        f9  = self._to_map(hs[9])
        f12 = self._to_map(hs[12])

        neck = self.aspp(self.fpn([f3, f6, f9, f12]))   # (B,256,Hp,Wp)

        d1 = self.dec1(self.up1(neck))     # 2× → (B,256,Hp*2,Wp*2)
        d2 = self.dec2(self.up2(d1))       # 4×
        d3 = self.dec3(self.up3(d2))       # 8×
        d4 = self.dec4(self.up4(d3))       # 16×

        # final bilinear to exact input resolution
        d4   = F.interpolate(d4, size=(H, W),
                             mode="bilinear", align_corners=False)
        main = self.seg_head(d4)           # (B, C, H, W)

        if self.training:
            a1 = F.interpolate(self.aux1(d1), size=(H, W),
                               mode="bilinear", align_corners=False)
            a2 = F.interpolate(self.aux2(d2), size=(H, W),
                               mode="bilinear", align_corners=False)
            a3 = F.interpolate(self.aux3(d3), size=(H, W),
                               mode="bilinear", align_corners=False)
            return main, a1, a2, a3

        return main

############################################################
# LOSSES
############################################################

class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=0, smooth=1e-6):
        super().__init__()
        self.C            = num_classes
        self.ignore_index = ignore_index
        self.smooth       = smooth

    def forward(self, logits, targets):
        logits  = logits.float()
        targets = targets.long()
        probs   = F.softmax(logits, dim=1)
        B, C, H, W = probs.shape
        one_hot = (F.one_hot(targets.clamp(0, C - 1), C)
                     .permute(0, 3, 1, 2).float())
        total, count = 0.0, 0
        for c in range(C):
            if c == self.ignore_index:
                continue
            p     = probs[:, c].reshape(B, -1)
            t     = one_hot[:, c].reshape(B, -1)
            inter = (p * t).sum(dim=1)
            union = p.sum(dim=1) + t.sum(dim=1)
            total += (1.0 - (2 * inter + self.smooth) /
                             (union    + self.smooth)).mean()
            count += 1
        return total / max(count, 1)


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sx = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        ).reshape(1, 1, 3, 3)
        sy = sx.permute(0, 1, 3, 2).contiguous()
        self.register_buffer("sobel_x", sx)
        self.register_buffer("sobel_y", sy)

    def forward(self, logits, targets):
        m  = targets.unsqueeze(1).float()
        sx = self.sobel_x.to(dtype=torch.float32, device=m.device)
        sy = self.sobel_y.to(dtype=torch.float32, device=m.device)
        ex     = F.conv2d(m, sx, padding=1)
        ey     = F.conv2d(m, sy, padding=1)
        edge   = (ex.abs() + ey.abs()).clamp(0, 1)
        weight = (1.0 + 4.0 * edge).squeeze(1)
        ce     = F.cross_entropy(logits.float(), targets.long(),
                                 ignore_index=0, reduction="none")
        return (ce * weight).mean()


class TotalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(ignore_index=0)
        self.dice = DiceLoss(cfg.NUM_CLASSES, ignore_index=0,
                             smooth=cfg.DICE_SMOOTH)
        self.bnd  = BoundaryLoss()

    def _single(self, logits, targets):
        l = logits.float()
        t = targets.long()
        return (cfg.CE_W   * self.ce(l, t) +
                cfg.DICE_W * self.dice(l, t) +
                cfg.EDGE_W * self.bnd(l, t))

    def forward(self, outputs, targets):
        if isinstance(outputs, (list, tuple)):
            main, a1, a2, a3 = outputs
            loss  =             self._single(main, targets)
            loss += cfg.AUX_W * self._single(a1,  targets)
            loss += cfg.AUX_W * self._single(a2,  targets)
            loss += cfg.AUX_W * self._single(a3,  targets)
            return loss
        return self._single(outputs, targets)

############################################################
# METRICS
############################################################

def compute_iou_from_cm(cm):
    ious = []
    for i in range(cm.shape[0]):
        tp    = cm[i, i]
        fp    = cm[:, i].sum() - tp
        fn    = cm[i, :].sum() - tp
        denom = tp + fp + fn
        ious.append(float(tp) / float(denom) if denom > 0 else 0.0)
    return float(np.mean(ious)), ious


def compute_dice_from_cm(cm):
    dices = []
    for i in range(cm.shape[0]):
        tp    = cm[i, i]
        fp    = cm[:, i].sum() - tp
        fn    = cm[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        dices.append(float(2 * tp) / float(denom) if denom > 0 else 0.0)
    return float(np.mean(dices)), dices


@torch.no_grad()
def validate(model, loader):
    model.eval()
    all_preds, all_targets = [], []

    for images, masks in loader:
        images = images.to(cfg.DEVICE)
        out    = model(images)
        preds  = torch.argmax(out, dim=1).cpu().numpy()
        masks  = masks.numpy()
        valid  = masks != 0
        all_preds.extend(  preds[valid].flatten())
        all_targets.extend(masks[valid].flatten())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)
    labels      = list(range(1, cfg.NUM_CLASSES))
    cm          = confusion_matrix(all_targets, all_preds, labels=labels)

    accuracy  = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, labels=labels,
                                average="macro", zero_division=0)
    recall    = recall_score(   all_targets, all_preds, labels=labels,
                                average="macro", zero_division=0)
    f1        = f1_score(       all_targets, all_preds, labels=labels,
                                average="macro", zero_division=0)
    miou,  class_iou  = compute_iou_from_cm(cm)
    mdice, class_dice = compute_dice_from_cm(cm)

    return dict(accuracy=accuracy, precision=precision,
                recall=recall, f1=f1,
                mIoU=miou, mDice=mdice,
                class_iou=class_iou, class_dice=class_dice, cm=cm)


def print_metrics(m, epoch=None):
    tag = f"Epoch {epoch}" if epoch else "Final"
    bar = "=" * 56
    print(f"\n{bar}\n  {tag} -- Validation Metrics\n{bar}")
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1 Score  : {m['f1']:.4f}")
    print(f"  mIoU      : {m['mIoU']:.4f}")
    print(f"  mDice     : {m['mDice']:.4f}")
    print(f"\n  {'Class':<14} {'IoU':>7}  {'Dice':>7}")
    print(f"  {'-'*32}")
    for i, name in enumerate(cfg.CLASS_NAMES[1:]):
        print(f"  {name:<14} {m['class_iou'][i]:>7.4f}  {m['class_dice'][i]:>7.4f}")
    print(f"\n  Confusion Matrix (classes 1-7):\n{m['cm']}")
    print(bar + "\n")

############################################################
# BUILD DATALOADERS
############################################################

train_imgs, train_masks = collect_split(
    os.path.join(cfg.DATA_ROOT, "Train", "Train"))
val_imgs,   val_masks   = collect_split(
    os.path.join(cfg.DATA_ROOT, "Val",   "Val"))

train_loader = DataLoader(
    LoveDADataset(train_imgs, train_masks,
                  processor=processor, augment=True),
    batch_size=cfg.BATCH_SIZE,
    shuffle=True,
    num_workers=cfg.NUM_WORKERS,
    pin_memory=True)

val_loader = DataLoader(
    LoveDADataset(val_imgs, val_masks,
                  processor=processor, augment=False),
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    num_workers=cfg.NUM_WORKERS,
    pin_memory=True)

############################################################
# BUILD MODEL
############################################################

model     = RADDINOSeg(encoder).to(cfg.DEVICE)
criterion = TotalLoss()

encoder_params = list(model.encoder.parameters())
decoder_params = [p for n, p in model.named_parameters()
                  if "encoder" not in n]

optimizer = torch.optim.AdamW([
    {"params": encoder_params, "lr": cfg.LR * 0.1},
    {"params": decoder_params, "lr": cfg.LR},
], weight_decay=cfg.WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.EPOCHS, eta_min=1e-7)

scaler = torch.cuda.amp.GradScaler(enabled=cfg.AMP)

total_p   = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters()
                if p.requires_grad)
print(f"Total params     : {total_p:,}")
print(f"Trainable params : {trainable:,}")

############################################################
# TRAINING LOOP
############################################################

best_miou = 0.0

for epoch in range(1, cfg.EPOCHS + 1):

    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}")

    for images, masks in pbar:
        images = images.to(cfg.DEVICE)
        masks  = masks.to(cfg.DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=cfg.AMP):
            outputs = model(images)

        # loss always in float32
        loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg = running_loss / len(train_loader)
    print(f"\nEpoch {epoch}  |  Loss: {avg:.4f}"
          f"  |  LR_enc: {optimizer.param_groups[0]['lr']:.2e}"
          f"  |  LR_dec: {optimizer.param_groups[1]['lr']:.2e}")

    metrics = validate(model, val_loader)
    print_metrics(metrics, epoch=epoch)
    scheduler.step()

    if metrics["mIoU"] > best_miou:
        best_miou = metrics["mIoU"]
        ckpt = os.path.join(cfg.CKPT_DIR, "best_raddino_loveda.pth")
        torch.save({
            "epoch":      epoch,
            "state_dict": model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "mIoU":       best_miou,
            "mDice":      metrics["mDice"]
        }, ckpt)
        print(f"  Best model saved  mIoU={best_miou:.4f}  → {ckpt}\n")

# ── final checkpoint ──────────────────────────────────────
torch.save(model.state_dict(),
           os.path.join(cfg.CKPT_DIR, "final_raddino_loveda.pth"))
print("Training complete.")
