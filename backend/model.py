import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import h5py
import io

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMG_SIZE    = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def _p(arr):
    return nn.Parameter(torch.tensor(np.array(arr), dtype=torch.float32))

def _load_bn(layer, grp):
    layer.weight             = _p(grp["weight"])
    layer.bias               = _p(grp["bias"])
    layer.running_mean       = torch.tensor(np.array(grp["running_mean"]), dtype=torch.float32)
    layer.running_var        = torch.tensor(np.array(grp["running_var"]),  dtype=torch.float32)
    layer.num_batches_tracked = torch.tensor(int(np.array(grp["num_batches_tracked"])))

def _load_conv(layer, grp):
    layer.weight = _p(grp["weight"])

def _load_linear(layer, grp):
    layer.weight = _p(grp["weight"])
    layer.bias   = _p(grp["bias"])


# ══════════════════════════════════════════════════════
#  MODEL 1 — ResNet50
# ══════════════════════════════════════════════════════
class ResNet50_FullTune(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=None)
        self.conv1=base.conv1; self.bn1=base.bn1; self.relu=base.relu
        self.maxpool=base.maxpool; self.layer1=base.layer1; self.layer2=base.layer2
        self.layer3=base.layer3; self.layer4=base.layer4; self.avgpool=base.avgpool
        self.fc = nn.Sequential(
            nn.Linear(2048,512), nn.BatchNorm1d(512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512,256),  nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256,4),
        )
    def forward(self, x):
        x=self.conv1(x); x=self.bn1(x); x=self.relu(x); x=self.maxpool(x)
        x=self.layer1(x); x=self.layer2(x); x=self.layer3(x); x=self.layer4(x)
        x=self.avgpool(x); x=torch.flatten(x,1); return self.fc(x)

def _load_bottleneck(block, grp):
    _load_conv(block.conv1,grp["conv1"]); _load_bn(block.bn1,grp["bn1"])
    _load_conv(block.conv2,grp["conv2"]); _load_bn(block.bn2,grp["bn2"])
    _load_conv(block.conv3,grp["conv3"]); _load_bn(block.bn3,grp["bn3"])
    if "downsample" in grp:
        _load_conv(block.downsample[0],grp["downsample"]["0"])
        _load_bn(block.downsample[1],grp["downsample"]["1"])

def load_resnet50(path):
    model = ResNet50_FullTune()
    with h5py.File(path,"r") as f:
        w=f["weights"]
        _load_conv(model.conv1,w["conv1"]); _load_bn(model.bn1,w["bn1"])
        for ln in ["layer1","layer2","layer3","layer4"]:
            for idx in w[ln].keys():
                _load_bottleneck(getattr(model,ln)[int(idx)],w[ln][idx])
        fc=w["fc"]
        _load_linear(model.fc[0],fc["0"]); _load_bn(model.fc[1],fc["1"])
        _load_linear(model.fc[4],fc["4"]); _load_bn(model.fc[5],fc["5"])
        _load_linear(model.fc[8],fc["8"])
    model.eval(); print("✅ ResNet50 loaded"); return model


# ══════════════════════════════════════════════════════
#  MODEL 2 — VGG16
# ══════════════════════════════════════════════════════
class VGG16_BNHead(nn.Module):
    def __init__(self):
        super().__init__()
        base=models.vgg16(weights=None)
        self.features=base.features; self.avgpool=base.avgpool
        self.classifier=nn.Sequential(
            nn.Linear(25088,4096), nn.BatchNorm1d(4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096,1024),  nn.BatchNorm1d(1024), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(1024,256),   nn.BatchNorm1d(256),  nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256,4),
        )
    def forward(self, x):
        x=self.features(x); x=self.avgpool(x); x=torch.flatten(x,1)
        return self.classifier(x)

def load_vgg16(path):
    model = VGG16_BNHead()
    with h5py.File(path,"r") as f:
        feat=f["weights"]["features"]
        for idx in feat.keys():
            layer=model.features[int(idx)]
            if isinstance(layer,nn.Conv2d):
                w=np.array(feat[idx]["weight"])
                if w.shape!=tuple(layer.weight.shape): w=np.transpose(w,(0,3,1,2))
                layer.weight=_p(w); layer.bias=_p(feat[idx]["bias"])
        cls=f["weights"]["classifier"]
        for h,s in {"0":0,"4":4,"8":8,"12":12}.items():
            if h in cls: _load_linear(model.classifier[s],cls[h])
        for h,s in {"1":1,"5":5,"9":9}.items():
            if h in cls: _load_bn(model.classifier[s],cls[h])
    model.eval(); print("✅ VGG16 loaded"); return model


# ══════════════════════════════════════════════════════
#  MODEL 3 — DenseNet121
# ══════════════════════════════════════════════════════
class DenseNet121_Head(nn.Module):
    def __init__(self):
        super().__init__()
        base=models.densenet121(weights=None)
        self.features=base.features
        self.classifier=nn.Sequential(
            nn.Linear(1024,512), nn.BatchNorm1d(512), nn.ReLU(True), nn.Dropout(0.4),
            nn.Linear(512,256),  nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(256,4),
        )
    def forward(self, x):
        x=self.features(x)
        x=torch.nn.functional.relu(x,inplace=True)
        x=torch.nn.functional.adaptive_avg_pool2d(x,(1,1))
        x=torch.flatten(x,1); return self.classifier(x)

def _load_dense_layer(layer, grp):
    for key in grp.keys():
        attr=getattr(layer,key,None)
        if attr is None: continue
        if isinstance(attr,nn.Conv2d) and "weight" in grp[key]:
            attr.weight=_p(grp[key]["weight"])
        elif isinstance(attr,nn.BatchNorm2d):
            _load_bn(attr,grp[key])

def load_densenet121(path):
    model=DenseNet121_Head()
    with h5py.File(path,"r") as f:
        w=f["weights"]
        if "features" in w:
            feats=w["features"]
            if "conv0" in feats: model.features.conv0.weight=_p(feats["conv0"]["weight"])
            if "norm0" in feats: _load_bn(model.features.norm0,feats["norm0"])
            for key in feats.keys():
                if key.startswith("denseblock"):
                    blk=getattr(model.features,key,None)
                    if blk is None: continue
                    for lk in feats[key].keys():
                        lobj=getattr(blk,lk,None)
                        if lobj: _load_dense_layer(lobj,feats[key][lk])
                elif key.startswith("transition"):
                    tr=getattr(model.features,key,None)
                    if tr is None: continue
                    if "norm" in feats[key]: _load_bn(tr.norm,feats[key]["norm"])
                    if "conv" in feats[key] and hasattr(tr,"conv"):
                        tr.conv.weight=_p(feats[key]["conv"]["weight"])
                elif key=="norm5": _load_bn(model.features.norm5,feats["norm5"])
        if "classifier" in w:
            cls=w["classifier"]
            for h,s in {"0":0,"4":4,"8":8}.items():
                if h in cls: _load_linear(model.classifier[s],cls[h])
            for h,s in {"1":1,"5":5}.items():
                if h in cls: _load_bn(model.classifier[s],cls[h])
    model.eval(); print("✅ DenseNet121 loaded"); return model


# ══════════════════════════════════════════════════════
#  LOAD ALL 3 MODELS
# ══════════════════════════════════════════════════════
MODELS = {}

def load_all():
    configs = [
        ("ResNet50",    "brain_tumor_resnet50_fulltune.h5",    load_resnet50),
        ("VGG16",       "brain_tumor_vgg16.h5",                load_vgg16),
        ("DenseNet121", "brain_tumor_densenet121_fulltune.h5",  load_densenet121),
    ]
    for name, path, loader in configs:
        try:
            MODELS[name] = loader(path)
        except FileNotFoundError:
            print(f"⚠️  {name}: '{path}' not found — skipping")
        except Exception as e:
            print(f"⚠️  {name} failed: {e}")

    if not MODELS:
        raise RuntimeError("No models loaded. Check .h5 files are in backend/ folder.")
    print(f"\n✅ Ensemble ready — {len(MODELS)} model(s): {list(MODELS.keys())}\n")

load_all()


# ══════════════════════════════════════════════════════
#  MAJORITY VOTE ENSEMBLE
# ══════════════════════════════════════════════════════
def predict(image_bytes: bytes) -> dict:
    img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    individual = {}
    votes      = [0, 0, 0, 0]

    with torch.no_grad():
        for name, model in MODELS.items():
            probs = torch.softmax(model(tensor), dim=1)[0].numpy()
            top   = int(np.argmax(probs))
            votes[top] += 1
            individual[name] = {
                "scores":     {CLASS_NAMES[i]: round(float(probs[i])*100, 1) for i in range(4)},
                "prediction": CLASS_NAMES[top],
                "confidence": round(float(probs[top])*100, 1),
            }
            print(f"  {name}: {CLASS_NAMES[top]} ({round(float(probs[top])*100,1)}%)")

    max_votes     = max(votes)
    final_class   = CLASS_NAMES[votes.index(max_votes)]

    agreed_confs = [individual[n]["confidence"] for n in individual if individual[n]["prediction"] == final_class]
    avg_conf     = round(sum(agreed_confs) / len(agreed_confs), 1)

    avg_scores = {
        cls: round(sum(individual[n]["scores"][cls] for n in individual) / len(individual), 1)
        for cls in CLASS_NAMES
    }

    print(f"  → ENSEMBLE: {final_class} ({max_votes}/{len(MODELS)} votes, {avg_conf}% avg conf)\n")

    return {
        "prediction":   final_class,
        "confidence":   avg_conf,
        "votes":        max_votes,
        "total_models": len(MODELS),
        "scores":       avg_scores,
        "individual":   individual,
    }