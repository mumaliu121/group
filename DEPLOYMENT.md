# Staging / Prod 部署流程

## 环境说明

| 环境 | 分支   | 触发方式          | 说明                   |
|-----|-------|------------------|------------------------|
| dev | dev   | 手动推送         | 开发调试               |
| staging | main | PR 合并后自动触发 CI | 集成测试环境         |
| prod  | main  | 手动合并 + 人工审批   | 生产环境             |

---

## Staging 部署（自动）

```bash
# 1. CI 流水线自动在 push to main 时构建镜像
git checkout main
git merge dev
git push origin main

# 2. 查看流水线状态
gh run watch

# 3. 镜像推送到 registry
docker push registry.example.com/road-surface-classifier:main
```

---

## Prod 部署（手动）

```bash
# 1. 确认 staging 测试通过
# 2. 从 staging 创建生产 release tag
git tag -a v1.0.0 -m "生产发布 v1.0"
git push origin v1.0.0

# 3. 更新生产集群
kubectl set image deployment/road-surface \
  road-surface=registry.example.com/road-surface-classifier:v1.0.0

# 4. 灰度验证
kubectl rollout status deployment/road-surface
```

---

## 回滚

```bash
# 查看历史版本
kubectl rollout history deployment/road-surface

# 回滚到上一版本
kubectl rollout undo deployment/road-surface
```
