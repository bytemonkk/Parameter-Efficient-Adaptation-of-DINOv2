"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   DINOv2  +  CNN Decoder  +  LoRA Fine-tuning  ─  LoveDA Segmentation      ║
║   CUDA-optimised  |  AMP bf16  |  torch.compile  |  channels-last           ║
║   gradient-checkpointing  |  fused AdamW  |  persistent workers             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Stdlib
# ─────────────────────────────────────────────────────────────────────────────
import os
import math
import time
import warnings
import logging
from contextlib import nullcontext

# ─────────────────────────────────────────────────────────────────────────────
#  Third-party
# ─────────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler

from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  ██████╗ ███████╗ ██████╗
#  ██╔════╝██╔════╝██╔════╝
#  ██║     █████╗  ██║  ███╗
#  ██║     ██╔══╝  ██║   ██║
#  ╚██████╗██║     ╚██████╔╝
#   ╚═════╝╚═╝      ╚═════╝
# ─────────────────────────────────────────────────────────────────────────────

class CFG:
    # ── paths ──────────────────────────────────────────────────────────────
    DATA_ROOT   = r"C:\Users\Dell\Downloads\archive"
    CKPT_DIR    = "checkpoints_lora"

    # ── model ──────────────────────────────────────────────────────────────
    ENCODER     = "facebook/dinov2-base"
    IMAGE_SIZE  = 560           # must be divisible by patch_size (14) → 40 patches/side
    NUM_CLASSES = 8
    CLASS_NAMES = ["Background", "Building", "Road", "Water",
                   "Barren", "Forest", "Agriculture", "Other"]

    # ── LoRA ───────────────────────────────────────────────────────────────
    LORA_R      = 16            # rank
    LORA_ALPHA  = 32            # scaling = alpha / r
    LORA_DROP   = 0.05
    # target the Q and V projections in every attention layer
    LORA_TARGETS = ["query", "value"]

    # ── training ───────────────────────────────────────────────────────────
    EPOCHS      = 25
    BATCH_SIZE  = 4
    NUM_WORKERS = 4
    LR_ENC      = 5e-6         # LoRA adapters (encoder side)
    LR_DEC      = 5e-5         # CNN decoder
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 2
    GRAD_CLIP   = 1.0

    # ── loss weights ───────────────────────────────────────────────────────
    CE_W        = 0.50
    DICE_W      = 0.35
    EDGE_W      = 0.15
    AUX_W       = 0.40

    # ── CUDA / precision ───────────────────────────────────────────────────
    AMP         = True
    AMP_DTYPE   = torch.bfloat16   # safer than fp16 on Ampere+; no inf scaling
    COMPILE     = True              # torch.compile (requires PyTorch ≥ 2.0)
    CH_LAST     = True              # channels-last memory for CNN convolutions
    GRAD_CKPT   = True             # gradient checkpointing on encoder
    PIN_MEMORY  = True
    PERSISTENT_W = True            # persistent DataLoader workers

    # ── misc ───────────────────────────────────────────────────────────────
    SEED        = 42
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    DICE_SMOOTH = 1e-6

cfg = CFG()

# ─────────────────────────────────────────────────────────────────────────────
#  Reproducibility  &  CUDA tuning
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)

if cfg.DEVICE == "cuda":
    torch.cuda.manual_seed_all(cfg.SEED)
    torch.backends.cudnn.benchmark   = True   # fastest conv algo per shape
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True   # Ampere TF32 for matmuls
    torch.backends.cudnn.allow_tf32       = True

os.makedirs(cfg.CKPT_DIR, exist_ok=True)

log.info(f"Device        : {cfg.DEVICE}")
if cfg.DEVICE == "cuda":
    log.info(f"GPU           : {torch.cuda.get_device_name(0)}")
    log.info(f"VRAM          : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# ─────────────────────────────────────────────────────────────────────────────
#  ██████╗  █████╗ ████████╗ █████╗ ███████╗███████╗████████╗
#  ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝
#  ██║  ██║███████║   ██║   ███████║███████╗█████╗     ██║
#  ██║  ██║██╔══██║   ██║   ██╔══██║╚════██║██╔══╝     ██║
#  ██████╔╝██║  ██║   ██║   ██║  ██║███████║███████╗   ██║
#  ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝
# ─────────────────────────────────────────────────────────────────────────────

class LoveDADataset(Dataset):
    """
    Loads Rural + Urban splits for a given LoveDA root.
    Applies CLAHE → resize → normalise → optional augmentation.
    All heavy numpy ops stay on CPU; GPU tensors never leave the loader.
    """

    # ImageNet stats (DINOv2 pre-training)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, image_paths: list, mask_paths: list, augment: bool = False):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.augment     = augment
        self.clahe       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self) -> int:
        return len(self.image_paths)

    # ── CLAHE per-channel ─────────────────────────────────────────────────
    def _apply_clahe(self, img_rgb: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        lab[..., 0] = self.clahe.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # ── geometric augmentation ────────────────────────────────────────────
    def _augment(self, img: np.ndarray, mask: np.ndarray):
        if np.random.rand() > 0.5:
            img  = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        if np.random.rand() > 0.5:
            img  = np.flipud(img).copy()
            mask = np.flipud(mask).copy()
        k = np.random.randint(0, 4)
        img  = np.rot90(img,  k).copy()
        mask = np.rot90(mask, k).copy()

        # colour jitter in HSV (image only)
        if np.random.rand() > 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] *= np.random.uniform(0.8, 1.2)
            hsv[..., 2] *= np.random.uniform(0.8, 1.2)
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return img, mask

    def __getitem__(self, idx: int):
        # ── load ──────────────────────────────────────────────────────────
        img  = cv2.imread(self.image_paths[idx])
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # ── CLAHE ─────────────────────────────────────────────────────────
        img = self._apply_clahe(img)

        # ── resize ────────────────────────────────────────────────────────
        img  = cv2.resize(img,  (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
                          interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
                          interpolation=cv2.INTER_NEAREST)

        # ── augmentation ──────────────────────────────────────────────────
        if self.augment:
            img, mask = self._augment(img, mask)

        # ── normalise ─────────────────────────────────────────────────────
        img = img.astype(np.float32) / 255.0
        img = (img - self.MEAN) / self.STD

        # ── to tensor (contiguous channels-first) ─────────────────────────
        img  = torch.from_numpy(img.transpose(2, 0, 1))        # (3,H,W) float32
        mask = torch.from_numpy(mask.astype(np.int64)).clamp(0, cfg.NUM_CLASSES - 1)

        return img, mask


def collect_split(split_root: str):
    images, masks = [], []
    for domain in ("Rural", "Urban"):
        img_dir  = os.path.join(split_root, domain, "images_png")
        mask_dir = os.path.join(split_root, domain, "masks_png")
        for f in sorted(os.listdir(img_dir)):
            images.append(os.path.join(img_dir,  f))
            masks.append( os.path.join(mask_dir, f))
    return images, masks


def build_loaders():
    train_imgs, train_masks = collect_split(
        os.path.join(cfg.DATA_ROOT, "Train", "Train"))
    val_imgs,   val_masks   = collect_split(
        os.path.join(cfg.DATA_ROOT, "Val",   "Val"))

    loader_kw = dict(
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=(cfg.NUM_WORKERS > 0 and cfg.PERSISTENT_W),
        prefetch_factor=2 if cfg.NUM_WORKERS > 0 else None,
    )

    train_loader = DataLoader(
        LoveDADataset(train_imgs, train_masks, augment=True),
        batch_size=cfg.BATCH_SIZE, shuffle=True, **loader_kw)

    val_loader = DataLoader(
        LoveDADataset(val_imgs, val_masks, augment=False),
        batch_size=cfg.BATCH_SIZE, shuffle=False, **loader_kw)

    log.info(f"Train samples : {len(train_imgs):,}")
    log.info(f"Val   samples : {len(val_imgs):,}")
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
#  ██╗      ██████╗ ██████╗  █████╗
#  ██║     ██╔═══██╗██╔══██╗██╔══██╗
#  ██║     ██║   ██║██████╔╝███████║
#  ██║     ██║   ██║██╔══██╗██╔══██║
#  ███████╗╚██████╔╝██║  ██║██║  ██║
#  ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
# ─────────────────────────────────────────────────────────────────────────────

def build_lora_encoder() -> nn.Module:
    """
    Load DINOv2-Base and wrap every attention Q/V projection with a
    LoRA adapter (r=LORA_R).  All other parameters are frozen; only
    the LoRA A/B matrices and LayerNorm scales are trained.
    """
    log.info(f"Loading {cfg.ENCODER} …")
    encoder = AutoModel.from_pretrained(
        cfg.ENCODER, output_hidden_states=True)

    # gradient checkpointing  →  trade compute for VRAM
    if cfg.GRAD_CKPT:
        encoder.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r               = cfg.LORA_R,
        lora_alpha      = cfg.LORA_ALPHA,
        target_modules  = cfg.LORA_TARGETS,
        lora_dropout    = cfg.LORA_DROP,
        bias            = "none",
        # PEFT does not yet expose a TaskType for vision encoders;
        # FEATURE_EXTRACTION keeps all original outputs intact.
        task_type       = TaskType.FEATURE_EXTRACTION,
    )

    encoder = get_peft_model(encoder, lora_cfg)
    encoder.print_trainable_parameters()
    return encoder


# ─────────────────────────────────────────────────────────────────────────────
#  CNN  DECODER  BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Sequential):
    """Conv → BN → ReLU  (standard CNN cell)."""
    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, stride: int = 1,
                 padding: int = 1, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, dilation=dilation,
                      groups=groups, bias=bias),
            nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )


class BottleneckResBlock(nn.Module):
    """
    1×1 → 3×3 (optionally depthwise-separable) → 1×1 bottleneck
    with residual skip and optional dilation.
    """
    expansion = 4

    def __init__(self, in_ch: int, mid_ch: int, dilation: int = 1,
                 depthwise: bool = True):
        super().__init__()
        out_ch = mid_ch * self.expansion
        groups = mid_ch if depthwise else 1
        self.body = nn.Sequential(
            ConvBnRelu(in_ch,   mid_ch, kernel=1, padding=0),
            ConvBnRelu(mid_ch,  mid_ch, kernel=3,
                       padding=dilation, dilation=dilation,
                       groups=groups),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3))
            if in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.body(x) + self.skip(x))


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling — captures multi-scale context
    without downsampling.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 rates: tuple = (6, 12, 18, 24)):
        super().__init__()
        self.convs = nn.ModuleList([
            ConvBnRelu(in_ch, out_ch, kernel=1, padding=0),
            *[ConvBnRelu(in_ch, out_ch, kernel=3,
                         padding=r, dilation=r)
              for r in rates],
        ])
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )
        n_in = (1 + len(rates) + 1) * out_ch
        self.project = nn.Sequential(
            ConvBnRelu(n_in, out_ch, kernel=1, padding=0),
            nn.Dropout2d(p=0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        branches = [c(x) for c in self.convs]
        gap = F.interpolate(self.gap(x), size=(H, W),
                            mode="bilinear", align_corners=False)
        branches.append(gap)
        return self.project(torch.cat(branches, dim=1))


class FPNLateral(nn.Module):
    """
    Lightweight FPN that fuses 4 encoder levels into a single
    feature map at the coarsest resolution.
    """
    def __init__(self, enc_dim: int = 768, fpn_dim: int = 256):
        super().__init__()
        self.laterals = nn.ModuleList([
            ConvBnRelu(enc_dim, fpn_dim, kernel=1, padding=0)
            for _ in range(4)])
        # top-down smooth after merge
        self.smooth = nn.ModuleList([
            ConvBnRelu(fpn_dim, fpn_dim)
            for _ in range(3)])
        self.fuse = ConvBnRelu(fpn_dim * 4, fpn_dim * 2,
                               kernel=1, padding=0)

    def forward(self, feats: list) -> torch.Tensor:
        # feats: [f3, f6, f9, f12]  all (B, enc_dim, Hp, Wp)
        lats = [self.laterals[i](f) for i, f in enumerate(feats)]
        H, W = lats[0].shape[-2:]

        # top-down path
        for i in range(len(lats) - 2, -1, -1):
            upsampled = F.interpolate(lats[i + 1], size=lats[i].shape[-2:],
                                      mode="bilinear", align_corners=False)
            lats[i] = self.smooth[i](lats[i] + upsampled) \
                      if i < len(self.smooth) else lats[i] + upsampled

        # align all to coarsest (first) resolution and concatenate
        aligned = [F.interpolate(l, size=(H, W),
                                 mode="bilinear", align_corners=False)
                   for l in lats]
        return self.fuse(torch.cat(aligned, dim=1))


class DecoderBlock(nn.Module):
    """2× bilinear upsample + residual conv refinement."""
    def __init__(self, in_ch: int, out_ch: int,
                 skip_ch: int = 0, dilation: int = 1):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear",
                                align_corners=False)
        merged    = in_ch + skip_ch
        self.body = nn.Sequential(
            ConvBnRelu(merged, out_ch),
            BottleneckResBlock(out_ch, out_ch // 4,
                               dilation=dilation, depthwise=True),
        )

    def forward(self, x: torch.Tensor,
                skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.body(x)


class SegHead(nn.Module):
    """Prediction head: 3×3 conv → BN → ReLU → 1×1 conv."""
    def __init__(self, in_ch: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            ConvBnRelu(in_ch, in_ch),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(in_ch, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ─────────────────────────────────────────────────────────────────────────────
#  FULL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class DINOv2LoRACNNSeg(nn.Module):
    """
    DINOv2-Base (LoRA fine-tuned)  +  CNN Decoder  for LoveDA segmentation.

    Architecture
    ────────────
    Encoder  → ViT-B/14  (LoRA on Q, V)
    Neck     → FPN over layers {3, 6, 9, 12}  +  ASPP
    Decoder  → 4 × DecoderBlock (each 2× upsample)  +  final bilinear
    Heads    → 1 main  +  3 auxiliary (deep supervision)
    """

    PATCH_SIZE  = 14
    ENCODER_DIM = 768

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

        # ── auto-detect patch grid ────────────────────────────────────────
        with torch.no_grad():
            dummy = torch.zeros(
                1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE,
                device="cpu")
            hs = encoder(pixel_values=dummy,
                         output_hidden_states=True).hidden_states
            n_tok = hs[1].shape[1] - 1      # minus CLS
            self.Hp = self.Wp = int(math.isqrt(n_tok))

        log.info(f"Patch grid    : {self.Hp}×{self.Wp}  ({n_tok} patches)")

        # ── neck ─────────────────────────────────────────────────────────
        self.fpn  = FPNLateral(enc_dim=self.ENCODER_DIM, fpn_dim=256)
        self.aspp = ASPP(in_ch=512, out_ch=256, rates=(6, 12, 18, 24))

        # ── decoder ──────────────────────────────────────────────────────
        #  neck output → 256 ch  @  Hp×Wp  (≈ 40×40 for 560 px input)
        #  4 upsamples  →  ×2  ×4  ×8  ×16  (≈ 560 px)
        self.dec1 = DecoderBlock(256,  256, dilation=2)
        self.dec2 = DecoderBlock(256,  128, dilation=2)
        self.dec3 = DecoderBlock(128,   64, dilation=1)
        self.dec4 = DecoderBlock( 64,   32, dilation=1)

        # ── heads ─────────────────────────────────────────────────────────
        self.seg_head = SegHead(32,  cfg.NUM_CLASSES, dropout=0.1)
        self.aux1     = SegHead(256, cfg.NUM_CLASSES, dropout=0.1)
        self.aux2     = SegHead(128, cfg.NUM_CLASSES, dropout=0.1)
        self.aux3     = SegHead(64,  cfg.NUM_CLASSES, dropout=0.1)

        self._init_decoder_weights()

    # ── weight init ───────────────────────────────────────────────────────
    def _init_decoder_weights(self):
        for m in [self.fpn, self.aspp,
                  self.dec1, self.dec2, self.dec3, self.dec4,
                  self.seg_head, self.aux1, self.aux2, self.aux3]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        layer.weight, mode="fan_out",
                        nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    # ── token → spatial map ───────────────────────────────────────────────
    def _to_map(self, hidden: torch.Tensor) -> torch.Tensor:
        """(B, 1+N, C)  →  (B, C, Hp, Wp)"""
        x = hidden[:, 1:, :]                          # drop CLS
        B, _, C = x.shape
        return x.permute(0, 2, 1).reshape(B, C, self.Hp, self.Wp)

    # ── forward ───────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape

        # encoder — returns hidden_states tuple of length 13
        enc = self.encoder(pixel_values=x, output_hidden_states=True)
        hs  = enc.hidden_states          # [embed, L1, …, L12]

        f3  = self._to_map(hs[3])        # (B, 768, Hp, Wp)
        f6  = self._to_map(hs[6])
        f9  = self._to_map(hs[9])
        f12 = self._to_map(hs[12])

        neck = self.aspp(self.fpn([f3, f6, f9, f12]))   # (B, 256, Hp, Wp)

        d1 = self.dec1(neck)             # (B, 256,  Hp×2,  Wp×2)
        d2 = self.dec2(d1)              # (B, 128,  Hp×4,  Wp×4)
        d3 = self.dec3(d2)              # (B,  64,  Hp×8,  Wp×8)
        d4 = self.dec4(d3)              # (B,  32,  Hp×16, Wp×16)

        # final upsample to input resolution
        d4   = F.interpolate(d4, size=(H, W),
                             mode="bilinear", align_corners=False)
        main = self.seg_head(d4)         # (B, C, H, W)

        if self.training:
            up = lambda t: F.interpolate(
                t, size=(H, W), mode="bilinear", align_corners=False)
            return main, up(self.aux1(d1)), up(self.aux2(d2)), up(self.aux3(d3))

        return main


# ─────────────────────────────────────────────────────────────────────────────
#  ██╗      ██████╗ ███████╗███████╗
#  ██║     ██╔═══██╗██╔════╝██╔════╝
#  ██║     ██║   ██║███████╗███████╗
#  ██║     ██║   ██║╚════██║╚════██║
#  ███████╗╚██████╔╝███████║███████║
#  ╚══════╝ ╚═════╝ ╚══════╝╚══════╝
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, num_classes: int,
                 ignore_index: int = 0,
                 smooth: float = 1e-6):
        super().__init__()
        self.C            = num_classes
        self.ignore_index = ignore_index
        self.smooth       = smooth

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        logits  = logits.float()
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
    """
    Sobel-weighted cross-entropy — penalises boundary pixels more heavily.
    Buffers are registered so they move to the right device automatically.
    """
    def __init__(self):
        super().__init__()
        sx = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        ).reshape(1, 1, 3, 3)
        self.register_buffer("sobel_x", sx)
        self.register_buffer("sobel_y", sx.permute(0, 1, 3, 2).contiguous())

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        m  = targets.unsqueeze(1).float()
        ex   = F.conv2d(m, self.sobel_x, padding=1)
        ey   = F.conv2d(m, self.sobel_y, padding=1)
        edge = (ex.abs() + ey.abs()).clamp(0, 1)
        weight = (1.0 + 4.0 * edge).squeeze(1)
        ce = F.cross_entropy(logits.float(), targets.long(),
                             ignore_index=0, reduction="none")
        return (ce * weight).mean()


class SegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(ignore_index=0)
        self.dice = DiceLoss(cfg.NUM_CLASSES, ignore_index=0,
                             smooth=cfg.DICE_SMOOTH)
        self.bnd  = BoundaryLoss()

    def _single(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        l = logits.float()
        t = targets.long()
        return (cfg.CE_W   * self.ce(l, t)   +
                cfg.DICE_W * self.dice(l, t) +
                cfg.EDGE_W * self.bnd(l, t))

    def forward(self, outputs, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, (list, tuple)):
            main, a1, a2, a3 = outputs
            loss  =             self._single(main, targets)
            loss += cfg.AUX_W * self._single(a1,  targets)
            loss += cfg.AUX_W * self._single(a2,  targets)
            loss += cfg.AUX_W * self._single(a3,  targets)
            return loss
        return self._single(outputs, targets)


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────

def iou_dice_from_cm(cm: np.ndarray):
    ious, dices = [], []
    for i in range(cm.shape[0]):
        tp    = cm[i, i]
        fp    = cm[:, i].sum() - tp
        fn    = cm[i, :].sum() - tp
        ious.append(float(tp) / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0)
        dices.append(float(2 * tp) / (2 * tp + fp + fn)
                     if (2 * tp + fp + fn) > 0 else 0.0)
    return np.mean(ious), ious, np.mean(dices), dices


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    all_preds, all_targets = [], []

    for imgs, masks in loader:
        imgs = imgs.to(cfg.DEVICE, non_blocking=True)
        if cfg.CH_LAST:
            imgs = imgs.to(memory_format=torch.channels_last)
        with torch.autocast(cfg.DEVICE, dtype=cfg.AMP_DTYPE,
                            enabled=cfg.AMP):
            out = model(imgs)

        preds = torch.argmax(out, dim=1).cpu().numpy()
        masks = masks.numpy()
        valid = masks != 0
        all_preds.extend(  preds[valid].ravel())
        all_targets.extend(masks[valid].ravel())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)
    labels      = list(range(1, cfg.NUM_CLASSES))
    cm          = confusion_matrix(all_targets, all_preds, labels=labels)

    miou, c_iou, mdice, c_dice = iou_dice_from_cm(cm)

    return dict(
        accuracy  = accuracy_score( all_targets, all_preds),
        precision = precision_score(all_targets, all_preds,
                                    labels=labels, average="macro",
                                    zero_division=0),
        recall    = recall_score(   all_targets, all_preds,
                                    labels=labels, average="macro",
                                    zero_division=0),
        f1        = f1_score(       all_targets, all_preds,
                                    labels=labels, average="macro",
                                    zero_division=0),
        mIoU      = miou,
        mDice     = mdice,
        class_iou = c_iou,
        class_dice= c_dice,
        cm        = cm,
    )


def print_metrics(m: dict, epoch: int | None = None):
    tag = f"Epoch {epoch}" if epoch is not None else "Final"
    bar = "═" * 60
    print(f"\n{bar}")
    print(f"  {tag}  ─  Validation Metrics")
    print(bar)
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1 Score  : {m['f1']:.4f}")
    print(f"  mIoU      : {m['mIoU']:.4f}")
    print(f"  mDice     : {m['mDice']:.4f}")
    print(f"\n  {'Class':<16} {'IoU':>7}  {'Dice':>7}")
    print(f"  {'─'*34}")
    for i, name in enumerate(cfg.CLASS_NAMES[1:]):
        print(f"  {name:<16} {m['class_iou'][i]:>7.4f}  {m['class_dice'][i]:>7.4f}")
    print(f"\n  Confusion Matrix (fg classes 1–7):\n{m['cm']}")
    print(bar + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  SCHEDULER  (warmup cosine)
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosine(torch.optim.lr_scheduler.LRScheduler):
    """Linear warmup → cosine annealing, applied per-epoch."""

    def __init__(self, optimizer, warmup_epochs: int,
                 total_epochs: int, eta_min: float = 1e-7,
                 last_epoch: int = -1):
        self.warmup = warmup_epochs
        self.total  = total_epochs
        self.eta    = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup:
            scale = (e + 1) / max(self.warmup, 1)
        else:
            progress = (e - self.warmup) / max(self.total - self.warmup, 1)
            scale    = self.eta + 0.5 * (1.0 - self.eta) * \
                       (1.0 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]


# ─────────────────────────────────────────────────────────────────────────────
#  CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def save_ckpt(path: str, epoch: int, model: nn.Module,
              optimizer, scheduler, scaler, metrics: dict):
    # unwrap compile wrapper if present
    raw = getattr(model, "_orig_mod", model)
    torch.save({
        "epoch"    : epoch,
        "state_dict": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler"   : scaler.state_dict(),
        "mIoU"     : metrics["mIoU"],
        "mDice"    : metrics["mDice"],
    }, path)


def load_ckpt(path: str, model: nn.Module,
              optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location=cfg.DEVICE)
    raw  = getattr(model, "_orig_mod", model)
    raw.load_state_dict(ckpt["state_dict"])
    if optimizer  : optimizer.load_state_dict( ckpt["optimizer"])
    if scheduler  : scheduler.load_state_dict( ckpt["scheduler"])
    if scaler     : scaler.load_state_dict(    ckpt["scaler"])
    log.info(f"Resumed from {path}  (epoch {ckpt['epoch']},"
             f"  mIoU={ckpt['mIoU']:.4f})")
    return ckpt["epoch"], ckpt["mIoU"]


# ─────────────────────────────────────────────────────────────────────────────
#  ████████╗██████╗  █████╗ ██╗███╗   ██╗
#     ██║   ██╔══██╗██╔══██╗██║████╗  ██║
#     ██║   ██████╔╝███████║██║██╔██╗ ██║
#     ██║   ██╔══██╗██╔══██║██║██║╚██╗██║
#     ██║   ██║  ██║██║  ██║██║██║ ╚████║
#     ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders()

    # ── model ─────────────────────────────────────────────────────────────
    encoder = build_lora_encoder()
    model   = DINOv2LoRACNNSeg(encoder).to(cfg.DEVICE)

    # channels-last layout — faster on Ampere for CNN convolutions
    if cfg.CH_LAST and cfg.DEVICE == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # torch.compile  (mode="reduce-overhead" best for fixed-shape training)
    if cfg.COMPILE and cfg.DEVICE == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            log.info("torch.compile  ✓")
        except Exception as e:
            log.warning(f"torch.compile failed ({e}); running eager.")

    # ── print param counts ────────────────────────────────────────────────
    total_p   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    log.info(f"Total params     : {total_p:,}")
    log.info(f"Trainable params : {trainable:,}")
    log.info(f"Trainable %      : {100*trainable/total_p:.2f}%")

    # ── loss & optimiser ──────────────────────────────────────────────────
    criterion = SegLoss().to(cfg.DEVICE)

    raw   = getattr(model, "_orig_mod", model)   # unwrap compile
    enc_p = list(raw.encoder.parameters())
    dec_p = [p for n, p in raw.named_parameters() if "encoder" not in n]

    optimizer = torch.optim.AdamW(
        [
            {"params": enc_p, "lr": cfg.LR_ENC},
            {"params": dec_p, "lr": cfg.LR_DEC},
        ],
        weight_decay=cfg.WEIGHT_DECAY,
        fused=True if cfg.DEVICE == "cuda" else False,   # fused AdamW kernel
    )

    scheduler = WarmupCosine(
        optimizer,
        warmup_epochs=cfg.WARMUP_EPOCHS,
        total_epochs=cfg.EPOCHS,
        eta_min=1e-7,
    )

    # AMP scaler — bfloat16 does not need loss scaling but GradScaler
    # is a no-op when enabled=False, so we unify the codepath.
    use_scaler = (cfg.AMP and cfg.AMP_DTYPE == torch.float16)
    scaler     = GradScaler(enabled=use_scaler)

    autocast_ctx = lambda: torch.autocast(
        cfg.DEVICE, dtype=cfg.AMP_DTYPE, enabled=cfg.AMP)

    best_miou  = 0.0
    best_path  = os.path.join(cfg.CKPT_DIR, "best_dinov2_lora.pth")
    final_path = os.path.join(cfg.CKPT_DIR, "final_dinov2_lora.pth")

    # ── training loop ─────────────────────────────────────────────────────
    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        t0           = time.perf_counter()

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch:>3}/{cfg.EPOCHS}",
                    dynamic_ncols=True)

        for imgs, masks in pbar:
            # move to device — non_blocking with pin_memory is zero-copy
            imgs  = imgs.to(cfg.DEVICE, non_blocking=True)
            masks = masks.to(cfg.DEVICE, non_blocking=True)

            if cfg.CH_LAST and cfg.DEVICE == "cuda":
                imgs = imgs.to(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

            with autocast_ctx():
                outputs = model(imgs)

            # loss is always computed in float32
            loss = criterion(outputs, masks)

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.GRAD_CLIP)
                optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr_enc=f"{optimizer.param_groups[0]['lr']:.1e}",
                lr_dec=f"{optimizer.param_groups[1]['lr']:.1e}",
            )

        scheduler.step()

        avg    = running_loss / len(train_loader)
        dt     = time.perf_counter() - t0
        vram   = (torch.cuda.memory_reserved() / 1e9
                  if cfg.DEVICE == "cuda" else 0.0)

        log.info(
            f"Epoch {epoch:>3}  |  loss={avg:.4f}"
            f"  |  time={dt:.0f}s"
            f"  |  VRAM={vram:.1f}GB"
            f"  |  lr_enc={optimizer.param_groups[0]['lr']:.1e}"
            f"  |  lr_dec={optimizer.param_groups[1]['lr']:.1e}"
        )

        metrics = validate(model, val_loader)
        print_metrics(metrics, epoch=epoch)

        if metrics["mIoU"] > best_miou:
            best_miou = metrics["mIoU"]
            save_ckpt(best_path, epoch, model,
                      optimizer, scheduler, scaler, metrics)
            log.info(f"  ✦ Best saved  mIoU={best_miou:.4f}  →  {best_path}")

    # ── final save ────────────────────────────────────────────────────────
    save_ckpt(final_path, cfg.EPOCHS, model,
              optimizer, scheduler, scaler,
              {"mIoU": best_miou, "mDice": 0.0})
    log.info(f"Training complete.  Best mIoU={best_miou:.4f}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
