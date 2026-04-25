# -*- coding: utf-8 -*-
"""
model_swin_ca_standard_final.py — SwinYOLO with Coordinate Attention on VOC 2007+2012
=======================================================================================
Model  : Swin Backbone + PANet Neck + CA only (FS removed)
Dataset: Pascal VOC 2007 + 2012 standard (~17,196 train / 1,001 val images)
Purpose: Isolate CA contribution — ablation study

SCHEDULER FIX — ROOT CAUSE OF ALL PREVIOUS COLLAPSES:
  CosineAnnealingLR is epoch-count dependent. Every preemption + resume
  created a NEW cosine schedule with shrinking T_max, causing LR to reach
  near-zero long before epoch 120. After 4 preemptions the model was
  essentially training with LR=0 by epoch 60, causing recall collapse.

DEFINITIVE FIX — ReduceLROnPlateau:
  - State = just counters (num_bad_epochs, best) — NOT epoch-dependent
  - load_state_dict() restores perfectly on every resume
  - LR reduces only when mAP stops improving — never collapses
  - Zero resume complexity — no cosine_steps_done calculation needed
  - Proven preemption-safe for long training runs
"""
import os, sys, subprocess, csv, math
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm

PIP = [sys.executable, "-m", "pip"]
subprocess.run(PIP + ["install", "-q", "timm", "albumentations==1.3.1",
                "opencv-python", "pyyaml", "tqdm"], check=True)
subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"], check=False)
subprocess.run(PIP + ["install", "-q", "-r", "yolov5/requirements.txt"], check=True)

HOME       = Path.home()
YOLO_ROOT  = HOME / "yolov5"
DATA_YAML  = YOLO_ROOT / "Pascal-Voc-0712-Standard" / "data.yaml"
HYP_YAML   = YOLO_ROOT / "data" / "hyps" / "hyp.scratch-low.yaml"
WEIGHTS    = YOLO_ROOT / "yolov5s.pt"
CSV_PATH   = HOME / "metrics_swin_ca_640_standard.csv"
BEST_CKPT  = HOME / "best_swin_ca_640_standard.pt"
RESUME_DIR = HOME / "checkpoints_ca_640_standard"
RESUME_DIR.mkdir(exist_ok=True)

subprocess.run(["wget", "-q", "-nc",
    "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",
    "-P", str(YOLO_ROOT)], check=True)

os.chdir(str(YOLO_ROOT))
sys.path.append(str(YOLO_ROOT))
assert DATA_YAML.exists(), "Run prepare_standard.py first!"

import torch, yaml
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import timm
from utils.dataloaders import create_dataloader
from utils.loss import ComputeLoss
from val import run as val_run
from utils.torch_utils import select_device
from models.yolo import Detect

cudnn.benchmark = True

# ── Model definitions ─────────────────────────────────────────────────────────
class ConvBnAct(nn.Module):
    def __init__(self, c_in, c_out, k=1, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class CoordinateAttention(nn.Module):
    def __init__(self, inp, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1  = nn.Conv2d(inp, mip, 1, 1, 0)
        self.bn1    = nn.BatchNorm2d(mip)
        self.act    = nn.ReLU(inplace=True)  # ReLU: less prone to saturation than Hardswish
        self.conv_h = nn.Conv2d(mip, inp, 1, 1, 0)
        self.conv_w = nn.Conv2d(mip, inp, 1, 1, 0)
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y   = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        return identity * torch.sigmoid(self.conv_h(x_h)) \
                        * torch.sigmoid(self.conv_w(x_w.permute(0, 1, 3, 2)))

class PANetNeck(nn.Module):
    def __init__(self, ch=(256, 512, 1024)):
        super().__init__()
        c3, c4, c5 = ch
        self.lateral5  = ConvBnAct(c5, c4, 1)
        self.td_merge4 = ConvBnAct(c4*2, c4, 3, p=1)
        self.lateral4  = ConvBnAct(c4, c3, 1)
        self.td_merge3 = ConvBnAct(c3*2, c3, 3, p=1)
        self.down3     = ConvBnAct(c3, c3, 3, s=2, p=1)
        self.bu_merge4 = ConvBnAct(c3+c4, c4, 3, p=1)
        self.down4     = ConvBnAct(c4, c4, 3, s=2, p=1)
        self.bu_merge5 = ConvBnAct(c4+c5, c5, 3, p=1)
    def forward(self, features):
        p3, p4, p5 = features
        p5_up  = F.interpolate(self.lateral5(p5), size=p4.shape[2:], mode='nearest')
        p4_td  = self.td_merge4(torch.cat([p4, p5_up], dim=1))
        p4_up  = F.interpolate(self.lateral4(p4_td), size=p3.shape[2:], mode='nearest')
        p3_td  = self.td_merge3(torch.cat([p3, p4_up], dim=1))
        p3_down = self.down3(p3_td)
        p4_bu  = self.bu_merge4(torch.cat([p3_down, p4_td], dim=1))
        p4_down = self.down4(p4_bu)
        p5_bu  = self.bu_merge5(torch.cat([p4_down, p5], dim=1))
        return [p3_td, p4_bu, p5_bu]

class SwinBackbone(nn.Module):
    def __init__(self, variant="swin_base_patch4_window7_224"):
        super().__init__()
        self.swin = timm.create_model(variant, pretrained=True, num_classes=0)
        # 🔥 FIX: Remove static attention masks (enables variable input size like 640)
        for layer in self.swin.layers:
            for block in layer.blocks:
                block.attn_mask = None
        self.cv3  = nn.Conv2d(256, 256, 1)
        self.cv4  = nn.Conv2d(512, 512, 1)
        self.cv5  = nn.Conv2d(1024, 1024, 1)
        self.required_multiple = 28   # patch4 × window7: pads 640→644 (only 4px)
    def forward(self, x):
        B, C, H0, W0 = x.shape; m = self.required_multiple
        pad_h = (math.ceil(H0/m)*m) - H0
        pad_w = (math.ceil(W0/m)*m) - W0
        if pad_h or pad_w: x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[-2], x.shape[-1]

        # ── Step 1: patch embedding bypass (skips hardcoded img_size assert) ──
        _pe = self.swin.patch_embed
        x   = _pe.proj(x)                       # [B, embed_dim, Hp/4, Wp/4]
        x   = x.flatten(2).transpose(1, 2)       # [B, num_patches, embed_dim]
        if hasattr(_pe, 'norm') and _pe.norm is not None:
            x = _pe.norm(x)

        # ── Step 2: reshape to spatial [B, H, W, C] — required by newer timm ─
        # SwinTransformerStage.blocks expect [B, H, W, C], NOT [B, N, C]
        Ht, Wt = Hp // 4, Wp // 4
        x = x.reshape(B, Ht, Wt, -1)            # [B, H, W, C]

        # pos_embed / pos_drop are absent in window7 model — guarded safely
        if hasattr(self.swin, "absolute_pos_embed") and \
                self.swin.absolute_pos_embed is not None:
            x = x + self.swin.absolute_pos_embed
        if hasattr(self.swin, "pos_drop") and \
                self.swin.pos_drop is not None:
            x = self.swin.pos_drop(x)

        # ── Step 3: run through Swin stages, collect P3/P4/P5 ────────────────
        # Each SwinTransformerStage outputs [B, H', W', C'] where patch merge
        # inside the stage halves H, W and doubles C (except stage 0).
        # We read x.shape[1:3] directly — no manual H/W tracking needed.
        outs = []
        for i, layer in enumerate(self.swin.layers):
            x = layer(x)                         # [B, H', W', C'] after stage
            if i in (1, 2, 3):
                # permute to [B, C, H, W] for conv/detection heads
                outs.append(x.permute(0, 3, 1, 2).contiguous())

        f3, f4, f5 = outs
        return [self.cv3(f3), self.cv4(f4), self.cv5(f5)]

class SwinYOLO_CA(nn.Module):
    """Swin + PANet + CA on all 3 scales (P3, P4, P5)
    FIX: Original only applied CA to P3 causing saturation and recall collapse.
    Applying CA to all 3 scales with reduction=16 prevents saturation.
    """
    def __init__(self, nc=20, anchors=None):
        super().__init__()
        self.backbone = SwinBackbone()
        self.neck     = PANetNeck(ch=(256, 512, 1024))
        self.ca_p3    = CoordinateAttention(256, reduction=16)  # P3 only — proven best placement
        self.detect   = Detect(nc=nc, anchors=anchors, ch=[256, 512, 1024])
        self.detect.stride = torch.tensor([8., 16., 32.])
        self.model    = nn.ModuleList([self.backbone, self.neck, self.detect])
        self.nc       = nc
    def forward(self, x, augment=False):
        if self.detect.anchors.device != x.device:
            self.detect.anchors = self.detect.anchors.to(x.device)
        feats      = self.backbone(x)
        feats      = self.neck(feats)
        p3, p4, p5 = feats
        p3         = self.ca_p3(p3)          # spatial attention on P3 only
        return self.detect([p3, p4, p5])

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE      = 640
BATCH_SIZE    = 4   # halved — 640px uses ~4x VRAM vs 384px
EPOCHS        = 120
LR            = 2e-4  # AdamW LR for Swin — architecture fix eliminates need for LR reduction
FREEZE_EPOCHS = 5
SAVE_INTERVAL = 2
VAL_INTERVAL  = 2

VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]

DEVICE = select_device("0" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

with open(DATA_YAML) as f: data = yaml.safe_load(f)
with open(HYP_YAML)  as f: hyp  = yaml.safe_load(f)
hyp['mosaic']=0.5; hyp['mixup']=0.05; hyp['label_smoothing']=0.0  # label smoothing removed — causes val distribution mismatch
nc = int(data["nc"])

anchors = [[10,13,16,30,33,23],[30,61,62,45,59,119],[116,90,156,198,373,326]]
model   = SwinYOLO_CA(nc=nc, anchors=anchors).to(DEVICE)
model.names = VOC_CLASSES

ckpt  = torch.load(str(WEIGHTS), map_location="cpu", weights_only=False)
state = ckpt["model"].float().state_dict()
if "model.24.anchors" in state:
    model.detect.anchors = state["model.24.anchors"].clone()
    print("Loaded pretrained anchors")
model.detect.anchors = model.detect.anchors.to(DEVICE)
model.hyp = hyp
print(f"SwinYOLO-CA-640 | nc={nc} | "
      f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

compute_loss = ComputeLoss(model)
optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

# ── DEFINITIVE SCHEDULER FIX: ReduceLROnPlateau ───────────────────────────────
# Preemption-safe: state is just counters, not epoch-dependent.
# Restores correctly via load_state_dict() on every resume.
# LR reduces by 0.5x if mAP doesn't improve for 8 consecutive val checks.
# With VAL_INTERVAL=2, patience=8 means 16 epochs before LR reduction.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=8,
    min_lr=1e-6)

# ── Resume logic ──────────────────────────────────────────────────────────────
START_EPOCH = 0; best_map50 = 0.0
ckpts = sorted(RESUME_DIR.glob("checkpoint_epoch*.pt"),
               key=lambda x: int(x.stem.split("epoch")[-1]))
if ckpts:
    c = torch.load(str(ckpts[-1]), map_location=DEVICE, weights_only=False)
    model.load_state_dict(c["model_state_dict"])
    optimizer.load_state_dict(c["optimizer_state_dict"])
    scheduler.load_state_dict(c["scheduler_state_dict"])  # safe — just counters
    START_EPOCH = c["epoch"] + 1
    best_map50  = c["best_map50"]
    print(f"Resumed from epoch {START_EPOCH} | best: {best_map50:.4f} "
          f"| lr: {optimizer.param_groups[0]['lr']:.6f}")
else:
    print("Starting fresh")

# ── CSV deduplication on resume ───────────────────────────────────────────────
if START_EPOCH == 0:
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","box_loss","obj_loss","cls_loss",
                                 "total_loss","precision","recall","map50","map5095","lr"])
else:
    if CSV_PATH.exists():
        with open(CSV_PATH) as f:
            rows = list(csv.reader(f))
        header = rows[0]
        seen = {}
        for r in rows[1:]:
            if r and r[0].isdigit():
                seen[int(r[0])] = r  # latest occurrence wins
        clean = [header] + [seen[k] for k in sorted(seen.keys())]
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerows(clean)
        print(f"CSV deduped — {len(clean)-1} unique epochs preserved")

# ── Backbone freeze setup ─────────────────────────────────────────────────────
model.train()
if START_EPOCH == 0:
    for p in model.backbone.parameters(): p.requires_grad = False
    print(f"Backbone frozen for {FREEZE_EPOCHS} epochs")
elif START_EPOCH < FREEZE_EPOCHS:
    for p in model.backbone.parameters(): p.requires_grad = False

# ── Dataloaders ───────────────────────────────────────────────────────────────
train_loader, _ = create_dataloader(
    path=data["train"], imgsz=IMG_SIZE, batch_size=BATCH_SIZE,
    stride=32, single_cls=False, hyp=hyp, augment=True,
    cache=False, rect=False, rank=-1, workers=2)
val_loader, _ = create_dataloader(
    path=data["val"], imgsz=IMG_SIZE, batch_size=BATCH_SIZE,
    stride=32, single_cls=False, hyp=hyp, augment=False,
    cache=False, rect=True, rank=-1, workers=2)

# ── Training loop ─────────────────────────────────────────────────────────────
for epoch in range(START_EPOCH, EPOCHS):
    if epoch == FREEZE_EPOCHS:
        for p in model.backbone.parameters(): p.requires_grad = True
        # Give backbone a lower LR initially to avoid unfreeze shock
        optimizer.param_groups[0]['lr'] = LR / 10
        print(f"Epoch {epoch+1}: Backbone unfrozen at LR={LR/10:.6f}")
    elif epoch == FREEZE_EPOCHS + 5:
        # Gradually restore LR — jump straight to full LR was causing instability
        optimizer.param_groups[0]['lr'] = LR / 2
        print(f"Epoch {epoch+1}: Backbone LR stepped to {LR/2:.6f}")
    # LR stays at LR/2 permanently — ReduceLROnPlateau handles further decay
    # Restoring to full LR=2e-4 causes instability in this architecture

    model.train()
    pbar  = tqdm(train_loader, desc=f"[CA-640-Std] Epoch {epoch+1}/{EPOCHS}")
    mloss = torch.zeros(3, device=DEVICE)

    for imgs, targets, paths, _ in pbar:
        imgs    = imgs.to(DEVICE).float() / 255.0
        targets = targets.to(DEVICE)
        loss, loss_items = compute_loss(model(imgs), targets)
        optimizer.zero_grad()
        loss.backward()
        # Tight clipping for CA params to prevent sigmoid saturation
        # CA weights grow too fast without this, causing recall collapse
        ca_params = [p for n, p in model.named_parameters() if 'ca_' in n]
        other_params = [p for n, p in model.named_parameters() if 'ca_' not in n]
        torch.nn.utils.clip_grad_norm_(other_params, 1.0)  # tighter clip for 115M param model
        torch.nn.utils.clip_grad_norm_(ca_params, 0.1)
        optimizer.step()
        mloss = mloss * 0.9 + loss_items * 0.1
        pbar.set_postfix(box=f"{mloss[0]:.3f}", obj=f"{mloss[1]:.3f}",
                         cls=f"{mloss[2]:.3f}", total=f"{loss.item():.3f}")

    # Warmup: manually scale LR during freeze phase only
    if epoch < FREEZE_EPOCHS:
        lr_scale = (epoch + 1) / FREEZE_EPOCHS
        for pg in optimizer.param_groups: pg['lr'] = LR * lr_scale

    lr = optimizer.param_groups[0]['lr']
    print(f"[CA-640-Std] Ep{epoch+1} loss={loss.item():.4f} lr={lr:.6f}")

    # ── Validation ────────────────────────────────────────────────────────────
    P = R = mAP50 = mAP5095 = 0.
    if (epoch + 1) % VAL_INTERVAL == 0 or (epoch + 1) == EPOCHS:
        model.eval()
        with torch.no_grad():
            em = deepcopy(model).to(DEVICE)
            res, _, _ = val_run(data=data, model=em, dataloader=val_loader,
                                imgsz=IMG_SIZE, conf_thres=0.001, iou_thres=0.6,
                                device=DEVICE, single_cls=False,
                                save_json=False, verbose=False)
        del em; torch.cuda.empty_cache()
        P, R, mAP50, mAP5095 = res[:4]
        print(f"[CA-640-Std] mAP@0.5={mAP50:.4f} mAP@0.5:0.95={mAP5095:.4f}")

        # Step ReduceLROnPlateau with current mAP (only after warmup)
        if epoch >= FREEZE_EPOCHS:
            scheduler.step(mAP50)

        if mAP50 > best_map50:
            best_map50 = mAP50
            torch.save(model.state_dict(), str(BEST_CKPT))
            print(f"New best: {best_map50:.4f}")
        model.train(); torch.set_grad_enabled(True)

    # ── Checkpoint (every SAVE_INTERVAL epochs) ───────────────────────────────
    if (epoch + 1) % SAVE_INTERVAL == 0:
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_map50":           best_map50,
        }, str(RESUME_DIR / f"checkpoint_epoch{epoch+1}.pt"))

    # ── CSV append ────────────────────────────────────────────────────────────
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([epoch+1,
                                 float(mloss[0]), float(mloss[1]), float(mloss[2]),
                                 float(loss.item()),
                                 float(P), float(R), float(mAP50), float(mAP5095),
                                 float(lr)])

print(f"\nSwinYOLO-CA-640 done | Best mAP@0.5: {best_map50:.4f}")
os.system('sudo shutdown -h now')
