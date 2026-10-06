# 路亚装备智能客服 Agent · Phase 0–5

本目录包含 Phase 0 的数据层、Phase 1 的自然语言库存查询、Phase 2 的空结果降级，以及 Phase 3 的 Hello-Agents 库存/知识工具路由。配置见 [PHASE1.md](PHASE1.md)，库存降级规则见 [PHASE2.md](PHASE2.md)，统一 Agent 使用方法见 [PHASE3.md](PHASE3.md)。下文保留各阶段的独立入口。

已加入 [Phase 3.1 检索增强与对比指南](PHASE3_1.md)：默认 BM25＋向量混合召回＋本地重排，MQE/HyDE 可选且默认关闭。`--rag-mode basic` 保留原基础检索，`evaluate_rag.py` 可做同题对照。没有新增记忆或前端。

已加入 [Phase 4 简化记忆](PHASE4.md)：命令行默认支持最近对话上下文，`/remember` 明确保存长期偏好，按 `--user-id` 隔离；`--no-memory` 复现独立查询。Phase 3.1 的“无记忆”描述指该阶段历史状态。

最新入口是 [Phase 5 网页工作台](PHASE5.md)：暖白松绿 Gradio 界面，聊天、真实库存候选、知识引用、记忆偏好与检索设置。仍为本机文字应用，不扩展多模态。

## 试玩 Phase 5 网页

```powershell
docker compose up -d
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe web_app.py
```

打开 <http://127.0.0.1:7860>。模型配置复用 `.env`；快捷问题只填入，不自动发送。默认网页偏好档案为 `web-local`，和 CLI 的 `local` 分开。启动、测试及本地会话边界见 [PHASE5.md](PHASE5.md)。

## 试玩当前 Phase 4 Agent

```powershell
docker compose up -d
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe index_knowledge.py
.\.venv\Scripts\python.exe luya_agent.py --interactive --trace
```

首次准备知识库时运行索引命令；资料不变时无需重复运行。可以问 `推荐本店100元以内钓鲈鱼的饵`，再问 `预算改成50元以内`，或咨询 `铅头钩搭配软饵为什么容易挂底？`。`--trace` 显示实际调用的是 `inventory`、`rag` 或两者，以及读取了多少轮上下文。`/help` 查看记忆命令；网页入口见上文。

## 单独试玩 Phase 2 库存工具

在本目录的 PowerShell 中执行（需要 Docker Desktop，LLM 配置复用本地 `.env`）：

```powershell
docker compose up -d
.\.venv\Scripts\python.exe inventory_tool.py --interactive
```

输入 `新手在浑水钓黑鱼，预算40元以内，不易挂底` 可以观察空结果后预算放宽至 48 元、超预算提醒以及替代商品。加 `--strict` 则不重试；加 `--no-budget-relaxation` 则不展示超预算替代品。每次输入独立查询，没有对话记忆。本阶段未接入 Agent、RAG 或前端。

## 目录内容

- `schema.sql`：商品表与鱼种别名表结构。
- `seed_data.sql`：项目提供的 31 条 mock 商品数据和别名。
- `docker-compose.yml`：本项目独立的 MariaDB、Qdrant 本地服务。
- `sync_to_qdrant.py`：把商品描述编码并同步到 Qdrant，结构化字段保存为 payload。
- `verify_query.py`：执行阶段 0 验收查询。

## 启动基础设施

需要 Docker Desktop 正在运行。在本目录执行：

```powershell
docker compose up -d
docker compose ps
```

MariaDB 首次启动时会自动执行 `schema.sql` 和 `seed_data.sql`。已有数据库卷不会重跑初始化脚本；如果该数据库还没有建表或导入种子，只手动执行一次：

```powershell
docker compose exec -T mariadb sh -c 'mariadb -uroot -p"$MARIADB_ROOT_PASSWORD" < /docker-entrypoint-initdb.d/01-schema.sql'
docker compose exec -T mariadb sh -c 'mariadb -uroot -p"$MARIADB_ROOT_PASSWORD" < /docker-entrypoint-initdb.d/02-seed.sql'
```

`seed_data.sql` 使用普通 `INSERT`，重复执行会重复添加商品，请勿对已经灌过种子的数据库再次运行。

Compose 默认数据库密码只用于本机开发。需要修改时，在 `.env` 中设置 `MARIADB_ROOT_PASSWORD`，并在运行 Python 脚本的终端将相同密码设为 `DB_PASSWORD`。

## 创建 Python 环境并运行

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python sync_to_qdrant.py
python verify_query.py
```

首次同步会下载 `BAAI/bge-small-zh-v1.5` embedding 模型，之后模型文件会缓存在本机。

脚本可通过环境变量配置：`DB_HOST`、`DB_PORT`、`DB_USER`、`DB_PASSWORD`、`DB_NAME`、`QDRANT_URL`、`QDRANT_COLLECTION` 和 `EMBEDDING_MODEL`。

验收查询中，鱼种、预算和库存由 Qdrant payload 条件过滤；商品描述通过 embedding 相似度排序。查询返回商品且全部满足硬过滤条件，即通过本阶段验收。
