#!/usr/bin/env python3
"""
路面状态图像分类训练脚本
- 基于 MobileNetV2 迁移学习
- 集成 MLflow 实验追踪与模型注册
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import mlflow
import mlflow.pytorch
import yaml

sys.path.insert(0, str(Path(__file__).parent / "registry"))
from data_pipeline import (
    get_class_weights,
    build_dataset,
    build_dataloader,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1, np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description="训练路面状态分类模型")
    parser.add_argument("--config", default="ml/configs/train_config.yaml")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 覆盖超参
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["train"]["lr"] = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = cfg["num_classes"]

    # ── 数据加载 ──────────────────────────────────────────────
    train_ds = build_dataset(args.data_root, is_train=True, cfg=cfg)
    class_counts = np.bincount(train_ds.targets)
    class_weights = torch.tensor(
        [len(train_ds.targets) / c for c in class_counts], dtype=torch.float32
    )
    sampler = WeightedRandomSampler(
        weights=[class_weights[t] for t in train_ds.targets],
        num_samples=len(train_ds.targets),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], sampler=sampler, num_workers=2
    )

    # ── 模型构建 ──────────────────────────────────────────────
    model = models.mobilenet_v2(pretrained=True)
    in_dim = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_dim, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    # ── MLflow 实验追踪 ───────────────────────────────────────
    exp_name = cfg["mlflow"]["experiment_name"]
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(exp_name)

    run_name = f"{cfg['mlflow']['run_name_prefix']}{cfg['model_name']}_lr{cfg['train']['lr']}_ep{cfg['train']['epochs']}"

    with mlflow.start_run(run_name=run_name) as run:
        # 记录超参
        mlflow.log_params({
            "model_name": cfg["model_name"],
            "pretrained": cfg["pretrained"],
            "input_size": cfg["input_size"],
            "epochs": cfg["train"]["epochs"],
            "batch_size": cfg["train"]["batch_size"],
            "lr": cfg["train"]["lr"],
            "weight_decay": cfg["train"]["weight_decay"],
            "num_classes": num_classes,
            "total_samples": len(train_ds),
        })

        # 记录类别权重
        mlflow.log_dict({"class_weights": class_weights.tolist()}, "class_weights.json")

        best_f1, best_epoch = 0.0, 0
        cm_best = None

        for epoch in range(1, cfg["train"]["epochs"] + 1):
            loss, macro_f1, preds, labels = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            cm = confusion_matrix(labels, preds)

            mlflow.log_metric("train/loss", loss, step=epoch)
            mlflow.log_metric("train/macro_f1", macro_f1, step=epoch)
            mlflow.log_metric("train/class_imbalance_ratio",
                              float(class_counts.max() / class_counts.min()), step=epoch)

            print(f"Epoch {epoch:>2}/{cfg['train']['epochs']}  "
                  f"Loss: {loss:.4f}  Macro-F1: {macro_f1:.4f}")

            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_epoch = epoch
                cm_best = cm
                # 保存模型 artifact
                model_path = f"models/model_epoch_{epoch}.pth"
                torch.save(model.state_dict(), model_path)
                mlflow.log_artifact(model_path, artifact_path="models")

        # 记录最终指标
        report = classification_report(labels, preds, output_dict=True, zero_division=0)
        mlflow.log_dict(report, "classification_report.json")
        np.save("confusion_matrix.npy", cm_best)
        mlflow.log_artifact("confusion_matrix.npy", artifact_path="metrics")

        # 注册模型到 MLflow Model Registry
        model_uri = mlflow.register_model(
            f"runs:/{run.info.run_id}/models",
            model_name="road-surface-classifier",
        )
        print(f"\n✅ 训练完成 | 最佳 Macro-F1: {best_f1:.4f} (Epoch {best_epoch})")
        print(f"📦 Run ID: {run.info.run_id}")
        print(f"🏷️  模型已注册: road-surface-classifier v{model_uri.version}")

        # 写入 current_model.md
        current_md = Path(__file__).parent / "current_model.md"
        current_md.write_text(
            f"# 当前生产模型\n\n"
            f"- **Run ID**: `{run.info.run_id}`\n"
            f"- **模型名称**: road-surface-classifier\n"
            f"- **注册版本**: v{model_uri.version}\n"
            f"- **最佳 Macro-F1**: {best_f1:.4f}\n"
            f"- **最佳 Epoch**: {best_epoch}\n"
            f"- **Git Commit**: `{os.popen('git rev-parse HEAD').read().strip()}`\n"
        )
        print(f"📝 current_model.md 已更新")


if __name__ == "__main__":
    main()
