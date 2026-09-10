# deepface-local-api

基于 [DeepFace](https://github.com/serengil/deepface) 的本地人脸识别库，使用**目录型人脸库**（`DeepFace.find` 的 `db_path`），并提供 FastAPI。

人脸目录结构（`InitFace` 按身份证号或工号建库）：

```text
{source}/IdentityCard/{身份证号}/*.jpg
{source}/JobNumber/{工号}/*.jpg
# 或扁平：
{source}/{身份证号或工号}/*.jpg
```

内部会整理成 DeepFace 目录库 `{db_path}/{identity}/{photo}.jpg`，并重建 embeddings pickle。

## 安装

```bash
pip install -e .
```

首次调用会下载 DeepFace 模型（识别 / 检测 / 表情），请保证能访问模型下载源。TensorFlow 2.16+ 需要 `tf-keras`（已写入依赖）。修改依赖后请**重启** API 进程。

## 作为 Python 库

```python
from deepface_local_api import FaceService, IdentityKey

svc = FaceService()
svc.init_face("./faces", identity_key=IdentityKey.identity_card)
svc.register("/abs/path/alice.jpg", identity="alice")
print(svc.verify("/abs/path/a.jpg", "/abs/path/b.jpg"))
print(svc.search("/abs/path/query.jpg", k=5))
print(svc.analyze_emotion("/abs/path/face.jpg"))
```

也可通过环境变量预设目录：`FACE_DB_PATH=/path/to/face_db`。

## 启动 API

```bash
python -m deepface_local_api --db-path ./face_db --port 8000
```

文档：`http://127.0.0.1:8000/docs`

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| POST | `/v1/faces/init` | 重建人脸 embedding `{ "source_path":"...","identity_key":"IdentityCard" }` |
| POST | `/v1/store/init` | 仅指定人脸目录 `{ "db_path": "..." }` |
| POST | `/v1/faces/register` | 注册 `{ "img_path": "...", "identity": "alice" }` |
| POST | `/v1/faces/verify` | 1:1 对比 `{ "img_path1": "...", "img_path2": "..." }` |
| POST | `/v1/faces/search` | 1:N 搜索 `{ "img_path": "..." }` |
| POST | `/v1/faces/emotion` | 表情分析 `{ "img_path": "..." }` |

示例：

```bash
curl -X POST http://127.0.0.1:8000/v1/faces/init \
  -H 'Content-Type: application/json' \
  -d '{"source_path":"/data/faces","identity_key":"IdentityCard"}'

curl -X POST http://127.0.0.1:8000/v1/store/init \
  -H 'Content-Type: application/json' \
  -d '{"db_path":"/tmp/face_db"}'

curl -X POST http://127.0.0.1:8000/v1/faces/register \
  -H 'Content-Type: application/json' \
  -d '{"img_path":"/tmp/alice.jpg","identity":"alice"}'

curl -X POST http://127.0.0.1:8000/v1/faces/verify \
  -H 'Content-Type: application/json' \
  -d '{"img_path1":"/tmp/a.jpg","img_path2":"/tmp/b.jpg"}'

curl -X POST http://127.0.0.1:8000/v1/faces/search \
  -H 'Content-Type: application/json' \
  -d '{"img_path":"/tmp/query.jpg"}'

curl -X POST http://127.0.0.1:8000/v1/faces/emotion \
  -H 'Content-Type: application/json' \
  -d '{"img_path":"/tmp/face.jpg"}'
```

## 配置

环境变量前缀为 `FACE_`：

- `FACE_DB_PATH`：默认人脸目录
- `FACE_MODEL_NAME`：默认 `VGG-Face`
- `FACE_DETECTOR_BACKEND`：默认 `opencv`
- `FACE_DISTANCE_METRIC`：默认 `cosine`
