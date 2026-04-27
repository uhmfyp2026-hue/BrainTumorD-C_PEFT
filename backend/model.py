import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import h5py
import io

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMG_SIZE = 224
MODEL_PATH = "brain_tumor_mobilenetv2_fulltune.h5"

# ── Exact architecture from inspect_model.py ──────────────────────────────────
# classifier[0]  Linear(25088 → 4096)
# classifier[1]  BatchNorm1d(4096)
# classifier[4]  Linear(4096 → 1024)
# classifier[5]  BatchNorm1d(1024)
# classifier[8]  Linear(1024 → 256)
# classifier[9]  BatchNorm1d(256)
# classifier[12] Linear(256 → 4)
# ─────────────────────────────────────────────────────────────────────────────

class MobileNetV2_PEFT_BNHead(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights=None)
        self.features = base.features
        self.avgpool  = base.avgpool  # 7x7 adaptive avg pool → (1280,7,7)

        # Custom classifier matching saved weights exactly
        # indices match the h5 keys: 0,1,_,_,4,5,_,_,8,9,_,_,12
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),       # [0]
            nn.BatchNorm1d(4096),         # [1]
            nn.ReLU(inplace=True),        # [2]  — no weights, skip in loader
            nn.Dropout(0.5),              # [3]  — no weights, skip in loader
            nn.Linear(4096, 1024),        # [4]
            nn.BatchNorm1d(1024),         # [5]
            nn.ReLU(inplace=True),        # [6]
            nn.Dropout(0.5),              # [7]
            nn.Linear(1024, 256),         # [8]
            nn.BatchNorm1d(256),          # [9]
            nn.ReLU(inplace=True),        # [10]
            nn.Dropout(0.5),              # [11]
            nn.Linear(256, 4),            # [12]
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _t(arr):
    """numpy array → float32 torch Parameter"""
    return nn.Parameter(torch.tensor(np.array(arr), dtype=torch.float32))


def load_weights(model, path):
    with h5py.File(path, "r") as f:
        feat = f["weights"]["features"]
        cls  = f["weights"]["classifier"]

        # ── features (conv layers) ────────────────────────────────────────────
        for idx_str in feat.keys():
            idx = int(idx_str)
            layer = model.features[idx]
            if isinstance(layer, nn.Conv2d):
                w = np.array(feat[idx_str]["weight"])
                # PyTorch Conv2d: (out, in, kH, kW)
                # h5 stored as:   (out, kH, kW, in)  ← need transpose
                if w.shape != tuple(layer.weight.shape):
                    w = np.transpose(w, (0, 3, 1, 2))
                layer.weight = _t(w)
                layer.bias   = _t(feat[idx_str]["bias"])

        # ── classifier ───────────────────────────────────────────────────────
        # Linear layers: [0], [4], [8], [12]
        linear_map = {0: 0, 4: 4, 8: 8, 12: 12}
        for h5_idx, seq_idx in linear_map.items():
            grp   = cls[str(h5_idx)]
            layer = model.classifier[seq_idx]
            if isinstance(layer, nn.Linear):
                layer.weight = _t(grp["weight"])
                layer.bias   = _t(grp["bias"])

        # BatchNorm layers: [1]→seq[1], [5]→seq[5], [9]→seq[9]
        bn_map = {1: 1, 5: 5, 9: 9}
        for h5_idx, seq_idx in bn_map.items():
            grp   = cls[str(h5_idx)]
            layer = model.classifier[seq_idx]
            if isinstance(layer, nn.BatchNorm1d):
                layer.weight       = _t(grp["weight"])        # gamma
                layer.bias         = _t(grp["bias"])          # beta
                layer.running_mean = torch.tensor(np.array(grp["running_mean"]), dtype=torch.float32)
                layer.running_var  = torch.tensor(np.array(grp["running_var"]),  dtype=torch.float32)
                layer.num_batches_tracked = torch.tensor(int(np.array(grp["num_batches_tracked"])))

    print("✅ All weights loaded successfully")
    return model


# ── Build & load ──────────────────────────────────────────────────────────────
model = MobileNetV2_BNHead()
model = load_weights(model, MODEL_PATH)
model.eval()
print(f"✅ Model ready — classes: {CLASS_NAMES}")

# ── Preprocessing (ImageNet stats, matches VGG16 training) ───────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def predict(image_bytes: bytes) -> dict:
    img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0)          # (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    preds = probs.numpy()
    print("Raw predictions:", {CLASS_NAMES[i]: round(float(preds[i])*100,1) for i in range(4)})

    scores = {CLASS_NAMES[i]: round(float(preds[i]) * 100, 1) for i in range(4)}
    top    = int(np.argmax(preds))
    return {
        "scores":     scores,
        "prediction": CLASS_NAMES[top],
        "confidence": round(float(preds[top]) * 100, 1)
    }