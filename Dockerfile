FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-musl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY pyproject.toml ./
RUN pip install --no-cache-dir ".[dev]" fastapi uvicorn

# 复制应用代码
COPY APP/ ./APP/
COPY ml/registry/ ./ml/registry/

# 加载模型（运行时从 volume 或环境变量指定路径）
ENV MODEL_PATH=models/model_epoch_30.pth
ENV PORT=8000

EXPOSE ${PORT}

CMD ["uvicorn", "APP.main:app", "--host", "0.0.0.0", "--port", "8000"]
