"""FastAPI 推理服务入口"""

import io
import os

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

app = FastAPI(title="路面状态分类服务", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 类别列表（与训练时一致）
CLASSES = [
    "dry-asphalt-smooth", "dry-asphalt-slight", "dry-asphalt-severe",
    "wet-asphalt-smooth", "wet-asphalt-slight", "wet-asphalt-severe",
    "water-asphalt-smooth", "water-asphalt-slight", "water-asphalt-severe",
    "dry-concrete-smooth", "dry-concrete-slight", "dry-concrete-severe",
    "wet-concrete-smooth", "wet-concrete-slight", "wet-concrete-severe",
    "water-concrete-smooth", "water-concrete-slight", "water-concrete-severe",
    "dry-gravel", "wet-gravel", "water-gravel",
    "dry-mud", "wet-mud", "water-mud",
    "fresh_snow", "melted_snow", "ice",
]

MODEL_PATH = os.getenv("MODEL_PATH", "models/model_epoch_30.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_transform_norm = (
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225],
)

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    from torchvision import models
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASSES))
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    _model = model
    return model


def _preprocess_image(img: Image.Image) -> torch.Tensor:
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(*_transform_norm),
    ])
    return preprocess(img).unsqueeze(0).to(DEVICE)


@app.get("/")
def root():
    return {"status": "ok", "service": "road-surface-classifier"}


@app.post("/predict")
def predict(file: UploadFile):
    """单图预测，返回概率最高的类别"""
    contents = file.file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="空文件")
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="无法解析图片格式")

    model = _load_model()
    tensor = _preprocess_image(img)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    return {
        "class": CLASSES[top_idx],
        "confidence": round(float(probs[top_idx]), 4),
        "all_classes": {cls: round(float(p), 4) for cls, p in zip(CLASSES, probs)},
    }


@app.post("/predict/batch")
def predict_batch(files: list[UploadFile]):
    """批量预测"""
    results = []
    for f in files:
        results.append(predict(f))
    return results
