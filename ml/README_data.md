# 数据集版本文档

## 数据源

路面状态图像数据集（Vali 20k），包含 27 类路面图像。

## 版本记录

### v1（当前）

- **数据路径**: `data/vali_20k/`
- **图片总数**: 19,860 张
- **类别数**: 27 类
- **日期范围**: 2022-01-25 ~ 2022-07-28
- **图像格式**: JPG，尺寸不统一（预处理时 Resize 到 224×224）

#### 类别分布

| 类别 | 样本数 |
|-----|-------|
| dry-asphalt-smooth | 820 |
| wet-asphalt-smooth | 820 |
| dry-concrete-severe | 820 |
| dry-concrete-slight | 820 |
| dry-concrete-smooth | 820 |
| wet-concrete-smooth | 820 |
| wet-concrete-slight | 820 |
| dry-asphalt-severe | 820 |
| dry-asphalt-slight | 820 |
| wet-asphalt-slight | 820 |
| wet-mud | 820 |
| water-mud | 820 |
| water-asphalt-smooth | 820 |
| water-asphalt-slight | 820 |
| water-concrete-smooth | 820 |
| fresh_snow | 820 |
| melted_snow | 820 |
| ice | 820 |
| water-concrete-severe | 820 |
| dry-mud | 820 |
| dry-gravel | 820 |
| water-gravel | 820 |
| wet-gravel | 820 |
| wet-asphalt-severe | 250 |
| water-asphalt-severe | 250 |
| wet-concrete-severe | 250 |
| water-concrete-slight | 250 |

#### 已知问题

- **类别不平衡**：4 个 severity 相关类别（wet/water-asphalt/concrete-severe & water-concrete-slight）仅有 250 张，为其他同类别的 30%
- 建议在训练时使用 `WeightedRandomSampler` 平衡样本权重（已在 `data_pipeline.py` 中实现）

### v2（规划）

- 计划扩充少量类别样本至平衡水平
- 增加数据增强（随机裁剪、旋转）的离线数据集版本

---

## DVC 使用

```bash
# 拉取数据
dvc fetch data.dvc
dvc checkout

# 查看数据状态
dvc status
```
