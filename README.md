# deepface-local-api

基于 [DeepFace](https://github.com/serengil/deepface) 的本地人脸识别服务，使用目录型人脸库（`DeepFace.find` 的 `db_path`），并提供 FastAPI。

人脸目录：

```text
face_db/
└── {userId}/
    └── {faceImageId}.jpg
```

例如：

```text
face_db/
├── 11/
│   ├── 8f31c2.jpg
│   ├── a92d17.jpg
│   └── c8210e.jpg
└── 25/
    ├── 7bd921.jpg
    └── 91ac32.jpg
```

## 项目结构

```text
src/deepface_local_api/
  config.py         # 环境变量 / 模型配置
  face_store.py     # FaceStore 目录库
  face_service.py   # FaceService 注册 / 搜索 / 表情
  app.py            # FastAPI
```

## 安装

```bash
pip install -e .
```

首次调用会下载 DeepFace 模型（识别 / 检测 / 表情）。TensorFlow 2.16+ 需要 `tf-keras`（已写入依赖）。修改依赖后请重启 API 进程。

注册会先裁剪人脸再写入 `{userId}/{faceImageId}.jpg`，并调用官方 `DeepFace.find(refresh_database=True)` 把新图同步进 embeddings pkl。搜索使用 `refresh_database=False`，只读已有索引。

## 作为 Python 库

```python
from deepface_local_api import FaceService

svc = FaceService("./face_db")
svc.register("/abs/path/alice.jpg", user_id="11")
print(svc.search("/abs/path/query.jpg"))
print(svc.emotion("/abs/path/face.jpg"))
```

## 启动 API

启动时必须传入人脸库目录：

```bash
python -m deepface_local_api --db-path ./face_db --port 8000
```

文档：`http://127.0.0.1:8000/docs`

统一返回：`{ "success": bool, "errMsg": string, "data": {} }`。搜索未命中时 `data` 为 `null`。

| 方法 | 路径 | 请求 | data |
| --- | --- | --- | --- |
| POST | `/register` | `{ "imagePath", "userId" }` | `{ "userId", "faceImageId", "imagePath" }`（裁剪后的图） |
| POST | `/search` | `{ "imagePath" }` | `{ "userId", "score", "cropImagePath" }` 或 `null`（`cropImagePath` 为库中裁剪图） |
| POST | `/emotion` | `{ "imagePath" }` | `{ "emotion" }`（angry / disgust / fear / happy / sad / surprise / neutral） |

```bash
curl -X POST http://127.0.0.1:8000/register \
  -H 'Content-Type: application/json' \
  -d '{"imagePath":"/tmp/alice.jpg","userId":"11"}'

curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"imagePath":"/tmp/query.jpg"}'

curl -X POST http://127.0.0.1:8000/emotion \
  -H 'Content-Type: application/json' \
  -d '{"imagePath":"/tmp/face.jpg"}'
```

## 配置

环境变量前缀为 `FACE_`：

- `FACE_DB_PATH`：人脸库目录（启动时由 `--db-path` 写入）
- `FACE_MODEL_NAME`：默认 `ArcFace`
- `FACE_DETECTOR_BACKEND`：默认 `opencv`
- `FACE_DISTANCE_METRIC`：默认 `cosine`
