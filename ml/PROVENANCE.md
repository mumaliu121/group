# 模型溯源（Provenance）

本文档记录每个模型版本与数据来源、代码版本的对应关系。

## 溯源字段说明

| 字段 | 说明 |
|-----|------|
| 模型 Run ID | MLflow 实验中的 run id |
| 数据版本 | DVC 管理的数据版本（见 README_data.md） |
| Git Commit | 训练时代码的 commit hash |
| 超参数快照 | 关键超参（lr、batch_size、epochs） |
| 创建时间 | ISO 8601 格式 |

## 模型版本记录

| 模型版本 | Run ID | 数据版本 | Git Commit | 创建时间 | 状态 |
|---------|--------|---------|-----------|---------|------|
| v1 | （待填写） | v1 | （待填写） | （待填写） | 生产 |

## 查询示例

```bash
# 通过 Run ID 查询 MLflow 实验详情
mlflow runs get <run-id> --tracking-uri mlruns/

# 查看 git 历史中该 commit 的改动
git show <commit-hash> --stat
```
