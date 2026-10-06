# Phase 5 · 野水路亚出钓工作台

为现有 Phase 0–4 Agent 添加 Gradio 文字网页，没有另建推荐逻辑，也没有扩展多模态、订单、付款或公网账号。

## 启动与试玩

在 `D:\project\agent\phase0_luya` 的 PowerShell 中执行：

```powershell
docker compose up -d
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe web_app.py
```

打开 <http://127.0.0.1:7860>。终端保持运行；按 Ctrl+C 停止。首次知识库准备仍按 Phase 3 执行 `index_knowledge.py`，已有索引且资料不变时无需重复索引，更不要重复导入商品种子。

模型和密钥沿用服务器本地 `.env`，不在网页中填写。启动只读取本地偏好，不要求调用 LLM；实际问题才调用 API。第一次库存/知识查询需要加载 embedding，混合知识检索还会加载重排模型，所以可能较慢。没有逐字 token 流式输出，页面先显示等待状态，再显示完整回答。

可以按顺序试：

1. 点击“新手选饵”：只填入问题，不会自动调用 API。点“发送”。
2. 问“预算改成50元以内”：沿用当前鱼种，重新查库存。
3. 问“铅头钩配软饵为什么容易挂底？只讲原理”：观察本轮知识来源。
4. 展开“长期偏好”，填写常钓鱼种、单件预算等，点“保存到本地偏好”。
5. 点“开启新对话”：清空短期上下文和候选，保留明确保存的偏好。
6. 清除偏好需要先勾选确认；清除后当前对话也清空，避免旧上下文恢复偏好。

聊天框仍可使用 `/help`、`/memory`、`/remember`、`/new`、`/forget`。网页没有 CLI 的 `exit` 退出功能；用浏览器关页或终端 Ctrl+C。

## 页面与后端

- 暖白/松绿配色，内联 SVG 山水和拟饵装饰，不加载外部图片、字体或追踪脚本；装饰并非商品实物照片。
- 独立展示对话、本轮库存候选、本轮知识引用和当前记忆条件。
- 商品名称、价格、库存来自 `inventory` 实际返回；替代候选明确保留差异和确认提示。页面不提供购买功能。
- 来源来自当前 `rag` 实际返回，非模型生成出处；只有 HTTPS 外部来源可点击，文件名仅作为资料标识。
- 默认 hybrid＋本地重排；basic 可对比。MQE/HyDE 默认关闭，开启后增加 API 调用；basic 不接受扩展。
- 严格库存模式与是否展示最多超预算 20% 的候选可调整；原预算不会被放宽后的预算覆盖。
- 检索记录显示实际工具、检索模式、重排执行情况和检索扩展调用数，不输出思考过程、密钥或原始供应商异常。

入口是 `web_app.py`；`web_backend.py` 处理状态与结果渲染；样式为 `assets/web.css`。原有命令行入口和工具不变。

## 会话与使用边界

网页默认使用固定的服务器本地档案 `web-local`，与 CLI 默认的 `local` 分开。可启动 `python web_app.py --user-id local` 共用原 CLI 偏好，或指定另一档案。**这个 ID 不是登录账号，不构成身份认证。**

同一网页服务的长期偏好由所有标签页共享；不同浏览器标签页的短期对话独立。偏好保存在已有 `.local/memory.sqlite3`，只有显式保存才持久化；普通对话在服务器内存中，最近 6 轮用于上下文，网页最多保留 20 轮展示。刷新/新开页面不是恢复聊天功能。浏览器状态配置 1 小时过期、最多 64 个状态；服务重启会丢失短期状态。

Gradio 的状态中只存 JSON 数据，不放 Agent、SQLite 连接或模型。每个回调独立创建/关闭 SQLite 连接，模型缓存复用；后端锁及共享队列串行处理请求（队列上限 8），适合本地演示，不是多用户生产服务。

只监听 `127.0.0.1`，关闭 share 公网链接与运行历史，项目目录禁止通过 Gradio 文件接口下载。不提供上传控件、模型连接配置或浏览器自选用户 ID；结果 HTML 转义，聊天 HTML 消毒。回调隐藏在公开 API 列表之外，但这不是鉴权。**不要直接用端口转发/反向代理公开部署**；公开服务仍需要认证、授权、配额、审计与隐私设计。SQLite 仍为本地明文偏好数据库，原 Phase 4 限制保持不变。

启动入口会仅为 localhost 增加代理绕过，保留外部请求的现有代理配置。端口冲突可用 `python web_app.py --port 7862`。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
# 浏览器测试依赖（非运行网页必需）
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe verify_web.py
# 真实调用现有 API，产生费用；需要 Docker 服务和已有模型/知识索引
.\.venv\Scripts\python.exe verify_web.py --with-llm
```

浏览器验收默认使用本机 Microsoft Edge 的无头模式，测试服务监听 7861，临时数据库不会修改你的长期偏好。截图保存到已忽略的 `.local/previews/`。无 `--with-llm` 时只测试快捷问题、命令、表单、确认与布局，不调用 LLM；加参数后额外验证真实库存推荐、预算追问与引用知识。

实现时核对了 [Gradio Blocks](https://www.gradio.app/docs/gradio/blocks)、[State](https://www.gradio.app/guides/state-in-blocks) 和 [自定义 CSS](https://www.gradio.app/guides/custom-CSS-and-JS)；Gradio 6 的主题与 CSS 在 launch 时传入。使用 OpenAI Docs 技能核对 [会话状态说明](https://developers.openai.com/api/docs/guides/conversation-state)，保持原有服务器端上下文及 API 适配器，本阶段不迁移 API。

### 本机验收记录（2026-10-03）

- 原有 113 项及新增 11 项测试，共 124 项通过；`pip check` 无依赖冲突。
- Microsoft Edge 实际页面验证：快捷填入、发送、本地命令、显式保存、删除确认和新对话；桌面及 390px 手机布局无横向溢出。
- 真实 API 浏览器链路：鲈鱼 100 元返回 5 件库存候选；追问预算改为 50 元，页面条件及候选更新；挂底知识问题展示本轮检索来源。
- 文件接口对 `.env` 及 `.local/memory.sqlite3` 下载请求返回 403；测试只读取响应状态，不输出文件内容。
