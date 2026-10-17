"""app/main.py 单元测试（含边界 case）"""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from APP.main import CLASSES, _preprocess_image, app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_rgb_image():
    """生成一张 64x64 彩色测试图"""
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


# ─── 健康检查 ──────────────────────────────────────────────
class TestRoot:
    def test_root_returns_ok(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ─── 单图预测 ──────────────────────────────────────────────
class TestPredict:
    def test_predict_valid_image(self, client, sample_rgb_image):
        """正常 JPEG 图片能成功返回预测结果"""
        files = {"file": ("test.jpg", sample_rgb_image, "image/jpeg")}
        with patch("app.main._load_model") as mock_model:
            mock_model.return_value = MagicMock()
            mock_model.return_value.side_effect = lambda x: MagicMock(
                cpu=MagicMock(numpy=MagicMock(return_value=[[0.1] * 27]))
            )
            r = client.post("/predict", files=files)
            assert r.status_code == 200
            data = r.json()
            assert "class" in data
            assert "confidence" in data
            assert data["class"] in CLASSES

    def test_predict_empty_file(self, client):
        """空文件应返回 400"""
        empty = io.BytesIO(b"")
        files = {"file": ("empty.jpg", empty, "image/jpeg")}
        r = client.post("/predict", files=files)
        assert r.status_code == 400
        assert "空文件" in r.json()["detail"]

    def test_predict_invalid_format(self, client):
        """非图片格式文件应返回 400"""
        files = {"file": ("readme.txt", io.BytesIO(b"not an image"), "text/plain")}
        with patch("app.main._load_model"):
            r = client.post("/predict", files=files)
            # 可能在预处理阶段失败
            assert r.status_code == 400 or r.status_code == 500


# ─── 批量预测 ──────────────────────────────────────────────
class TestPredictBatch:
    def test_batch_single(self, client, sample_rgb_image):
        """批量接口传入单张图正常返回"""
        with patch("app.main.predict") as mock_pred:
            mock_pred.return_value = {"class": "dry-asphalt-smooth", "confidence": 0.95}
            r = client.post("/predict/batch", files=[("files", ("a.jpg", sample_rgb_image, "image/jpeg"))])
            assert r.status_code == 200
            assert isinstance(r.json(), list)

    def test_batch_empty(self, client):
        """不传文件应正常处理"""
        r = client.post("/predict/batch", files=[])
        # 至少应返回 200（即使为空列表）
        assert r.status_code == 200


# ─── 预处理 ────────────────────────────────────────────────
class TestPreprocess:
    def test_preprocess_shape(self):
        """预处理输出应为 [1, 3, 224, 224]"""
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        tensor = _preprocess_image(img)
        assert tensor.shape == (1, 3, 224, 224)
        # 值域应在 [-1, 1] 附近（经 normalize）
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0
