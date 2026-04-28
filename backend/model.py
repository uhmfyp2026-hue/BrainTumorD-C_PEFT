import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import h5py
import io

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMG_SIZE    = 224
MODEL_PATH  = "brain_tumor_resnet50_fulltune.h5"

class ResNet50_FullTune(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=None)
        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def _p(arr):
    return nn.Parameter(torch.tensor(np.array(arr), dtype=torch.float32))

def _load_bn(layer, grp):
    layer.weight              = _p(grp["weight"])
    layer.bias                = _p(grp["bias"])
    layer.running_mean        = torch.tensor(np.array(grp["running_mean"]), dtype=torch.float32)
    layer.running_var         = torch.tensor(np.array(grp["running_var"]),  dtype=torch.float32)
    layer.num_batches_tracked = torch.tensor(int(np.array(grp["num_batches_tracked"])))

def _load_conv(layer, grp):
    layer.weight = _p(grp["weight"])

def _load_linear(layer, grp):
    layer.weight = _p(grp["weight"])
    layer.bias   = _p(grp["bias"])

def _load_bottleneck(block, grp):
    _load_conv(block.conv1, grp["conv1"])
    _load_bn(block.bn1,     grp["bn1"])
    _load_conv(block.conv2, grp["conv2"])
    _load_bn(block.bn2,     grp["bn2"])
    _load_conv(block.conv3, grp["conv3"])
    _load_bn(block.bn3,     grp["bn3"])
    if "downsample" in grp:
        ds = grp["downsample"]
        _load_conv(block.downsample[0], ds["0"])
        _load_bn(block.downsample[1],   ds["1"])

def load_weights(model, path):
    with h5py.File(path, "r") as f:
        w = f["weights"]
        _load_conv(model.conv1, w["conv1"])
        _load_bn(model.bn1,     w["bn1"])
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(model, layer_name)
            grp   = w[layer_name]
            for idx_str in grp.keys():
                _load_bottleneck(layer[int(idx_str)], grp[idx_str])
        fc_grp = w["fc"]
        _load_linear(model.fc[0], fc_grp["0"])
        _load_bn(model.fc[1],     fc_grp["1"])
        _load_linear(model.fc[4], fc_grp["4"])
        _load_bn(model.fc[5],     fc_grp["5"])
        _load_linear(model.fc[8], fc_grp["8"])
    print("✅ ResNet50 weights loaded successfully")
    return model

model = ResNet50_FullTune()
model = load_weights(model, MODEL_PATH)
model.eval()
print(f"✅ Model ready — classes: {CLASS_NAMES}")
print(f"   Architecture : ResNet50-FullFinetune")
print(f"   Val accuracy : 99.53%")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(image_bytes: bytes) -> dict:
    img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    preds  = probs.numpy()
    scores = {CLASS_NAMES[i]: round(float(preds[i]) * 100, 1) for i in range(4)}
    top    = int(np.argmax(preds))
    print("Prediction:", CLASS_NAMES[top], "| Confidence:", round(float(preds[top]) * 100, 1), "%")
    print("All scores:", scores)
    return {
        "scores":     scores,
        "prediction": CLASS_NAMES[top],
        "confidence": round(float(preds[top]) * 100, 1)
    }

