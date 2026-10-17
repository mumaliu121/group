"""
数据集预处理与类别权重计算模块
"""

import os
import numpy as np
import torch
from torchvision import datasets, transforms
from pathlib import Path


def get_class_weights(data_root: str, cfg: dict) -> torch.Tensor:
    """
    根据数据集中各类别样本数量，计算逆频率加权权重。
    用于处理 23 类平衡 / 4 类少量（250 vs 820）的类别不平衡问题。
    """
    ds = datasets.ImageFolder(root=data_root)
    targets = np.array(ds.targets)
    class_counts = np.bincount(targets)
    total = class_counts.sum()
    weights = torch.tensor(
        [total / c for c in class_counts], dtype=torch.float32
    )
    print(f"类别分布:\n{dict(zip(ds.classes, class_counts.tolist()))}")
    print(f"类别权重: {weights.tolist()}")
    return weights


def build_dataset(data_root: str, is_train: bool, cfg: dict) -> datasets.ImageFolder:
    """
    构建 ImageFolder 数据集，返回带 transform 的数据集对象。
    - is_train=True  : 加入水平翻转 + ColorJitter
    - is_train=False : 仅 Resize
    """
    size = cfg["input_size"]
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return datasets.ImageFolder(root=data_root, transform=transform)


def build_dataloader(dataset: datasets.ImageFolder, cfg: dict, is_train: bool = True):
    """
    构建 DataLoader：
    - 训练集使用 WeightedRandomSampler 处理类别不平衡
    - 验证集按自然顺序采样
    """
    if is_train:
        targets = np.array(dataset.targets)
        class_counts = np.bincount(targets)
        class_weights = torch.tensor(
            [len(targets) / c for c in class_counts], dtype=torch.float32
        )
        sample_weights = [class_weights[t].item() for t in targets]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=cfg["train"]["batch_size"],
            sampler=sampler, num_workers=2, pin_memory=True,
        )
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=cfg["train"]["batch_size"],
            shuffle=False, num_workers=2, pin_memory=True,
        )
