# MLflow 实验对比总结

## 实验配置

所有实验均在同一数据集（v1，19,860 张）上运行，使用 MobileNetV2 迁移学习。

## 实验记录

| 实验名 | 模型 | 学习率 | Epochs | Batch Size | 最佳 Macro-F1 | Run ID | 备注 |
|-------|------|-------|--------|-----------|--------------|--------|------|
| exp_mobilenet_v2_lr1e-4_ep30 | MobileNetV2 | 1e-4 | 30 | 32 | TBD | - | 基准实验 |
| exp_mobilenet_v2_lr5e-5_ep30 | MobileNetV2 | 5e-5 | 30 | 32 | TBD | - | 更低学习率 |
| exp_mobilenet_v2_lr1e-4_ep20 | MobileNetV2 | 1e-4 | 20 | 32 | TBD | - | 缩短训练 |

> 更新说明：每次训练结束后，将结果追加到本表并更新 `current_model.md`

## 评估指标说明

- **Macro-F1**：对 27 个类别等权重计算 F1，能有效反映少数类的表现
- **Confusion Matrix**：保存于 `mlruns/` 各 run 目录下的 `metrics/confusion_matrix.npy`

## 结论模板（训练完成后填写）

```
最佳实验: exp_mobilenet_v2_lr1e-4_ep30
原因: Macro-F1 最高，且验证集上各类别 F1 差异较小，泛化稳定
当前部署模型: road-surface-classifier v1
```
