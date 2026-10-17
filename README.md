# 路面状态分类项目

**项目名称**: road-surface-mlops  
**目标**: 基于图像的路面干湿状态与破损严重程度自动分类  
**类别数**: 27 类

---

## 项目结构

```
road-surface-mlops/
├── app/                          # 推理服务
│   ├── main.py                   # FastAPI 推理服务入口（端口 8000）
│   └── test_app.py               # ≥3 个功能单元测试，含边界 case
├── ml/                           # 机器学习模块
│   ├── train.py                  # 训练脚本（集成 MLflow 日志）
│   ├── registry/
│   │   ├── data_pipeline.py      # 数据集预处理、类别权重计算
│   │   └── configs/
│   │       └── train_config.yaml # 超参配置
│   ├── README_data.md            # DVC 数据集版本文档（v1/v2）
│   ├── README_experiments.md     # MLflow 实验对比总结
│   ├── current_model.md          # 当前生产使用的 run id、模型 artifact 路径
│   └── PROVENANCE.md             # 溯源：模型、数据、git commit 对应关系
├── data.dvc                      # DVC 数据指针文件，原始数据不提交 git
├── data/                         # 本地数据文件夹（gitignore，DVC 管理）
├── .github/workflows/ci.yml      # GitHub Actions CI/CD 流水线
├── Dockerfile                    # 推理服务镜像（slim 轻量化）
├── .dockerignore
├── .gitignore
├── pyproject.toml                # 依赖：torch、mlflow、dvc、fastapi、pytest 等
├── DEPLOYMENT.md                 # staging/prod 部署流程文档
└── README.md                     # 项目总览
```

---

## 数据集说明

- **总数**: 19,860 张 JPG 图片
- **日期范围**: 2022-01-25 ~ 2022-07-28
- **类别分布**: 23 类各 820 张，4 类各 250 张（存在类别不平衡，已通过 WeightedRandomSampler 缓解）

## 快速开始

### 安装依赖
```bash
pip install -e ".[dev]"
```

### 拉取数据（DVC）
```bash
dvc fetch data.dvc
dvc checkout
```

### 训练模型
```bash
cd ml && python train.py
```

### 启动推理服务
```bash
python app/main.py
# 访问 http://localhost:8000/docs
```

### 运行测试
```bash
pytest app/test_app.py -v
```

---

## 分支策略与流水线触发

| 分支 | 触发行为 |
|-----|---------|
| `main` | 每次 push → 运行 CI：lint + 单元测试 + 构建 Docker 镜像 |
| `dev` | 日常开发分支，PR 合并到 main 时触发完整 CI |
