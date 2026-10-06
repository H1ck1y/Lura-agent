# Phase 3：Hello-Agents 工具接入与知识问答

**后续更新：** 当前默认检索已升级为 [Phase 3.1](PHASE3_1.md) 的 BM25＋向量混合召回＋本地重排；MQE/HyDE 为默认关闭的可选实验。下文的“仅向量、无扩展”描述指原基础模式，可用 `--rag-mode basic` 复现；当前默认、分数含义及新增参数以 Phase 3.1 为准。

**Phase 4 更新：** 命令行默认已接入 [简化记忆](PHASE4.md)。下文“每次独立、没有记忆”是 Phase 3 历史行为，现用 `--no-memory` 复现；原 LuyaAgent 编程接口仍独立查询。

本阶段已把 Phase 2 的 `InventoryTool` 和只读 `RAGTool` 并列注册到 Hello-Agents `ToolRegistry`，由 `SimpleAgent` 内部工具调用循环完成路由和回答。**没有通过关键词 if/else 强制区分库存与知识**；路由能力通过真实模型验收，而不只靠模拟响应证明。

本阶段不实现记忆、长期偏好、Web 前端、下单、付款或物流。框架自带的聊天历史在每次提问开始/结束时清空，因此上一轮说“预算100”，下一轮说“再便宜一点”不会继承预算。请每次输入完整问题。

## 开始试玩

PowerShell（本地 `.env` 复用 Phase 1 的模型、密钥和地址）：

```powershell
cd D:\project\agent\phase0_luya
docker compose up -d
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 第一次建立知识索引；知识文件变更后再运行
.\.venv\Scripts\python.exe index_knowledge.py

# 新的统一入口，和 inventory_tool.py 不同
.\.venv\Scripts\python.exe luya_agent.py --interactive --trace
```

`--trace` 显示实际工具调用名称，不显示模型内部思考。终端中生成的是带解释的中文回答，知识答案附来源，库存答案附实际商品价格、库存与降级差异。

| 输入 | 预期工具 |
| --- | --- |
| `我是新手，钓鲈鱼，预算100元以内，推荐本店商品` | `inventory` |
| `铅头钩搭配软饵为什么容易挂底？只讲原理，不买东西` | `rag` |
| `清水和浑水怎么选择拟饵颜色？` | `rag` |
| `在浑水钓黑鱼，预算40元以内，推荐本店商品` | `inventory`，保留 Phase 2 超预算提醒 |
| `推荐本店100元以内钓鲈鱼的饵，并解释新手怎么练习匀速收线` | `inventory` 和 `rag` |
| `你好` | 无工具，说明服务范围 |

输入 `exit`、`quit`、`退出` 均可结束。模型/API 故障会明确显示错误，交互模式可继续下一次输入。

单次执行与结构化结果：

```powershell
.\.venv\Scripts\python.exe luya_agent.py --query "铅头钩配软饵为什么容易挂底？" --trace
.\.venv\Scripts\python.exe luya_agent.py --query "钓鲈鱼，预算100，推荐本店商品" --json
```

如果预算不能增加，使用 `--no-budget-relaxation`；任何库存硬条件都不能放宽时使用 `--strict-inventory`。两者仅影响库存工具，知识查询不受影响。原来的 `inventory_tool.py` 入口仍可独立使用，原 Phase 1/2 接口保持兼容。

## RAG 是怎样接进来的

按计划书“基础设施调包、业务逻辑自研”的范围：

- 使用 Hello-Agents 0.2.9 的 `SimpleAgent`、`ToolRegistry`、`RAGTool`、Markdown 文档分块器和向量存储写入方法。
- `KnowledgeRAGTool` 是框架 `RAGTool` 的只读适配子类：明确接入已有本地 BGE embedding 和 Qdrant，增加来源信息、版本过滤及服务错误处理。
- 未使用框架默认云 embedding，也未开启 MQE、HyDE 等额外模型调用；知识检索本身只调用本地 embedding，最终回答由 Agent 模型组织。
- 没有注册知识工具的 `add_document`、`add_text`、`clear`、任意文件路径或网络读取等动作。建索引只能通过独立管理入口 `index_knowledge.py`。

实现适配的原因：框架默认 RAG 会创建自身 embedding/LLM 客户端，部分检索错误会变成空列表。这里复用现有模型，并把服务失败与“资料不足”分开；本地 Qdrant 客户端不经过系统 HTTP 代理。没有修改 `.venv` 内的框架源码。

## 知识库与来源

`knowledge_base/sources.json` 是索引清单，初始包含 5 篇短中文入门资料：拟饵分类、铅头钩挂底、匀收/轻抽、清浑水选色、米诺操作。资料根据 [SHIMANO 入门指南](https://fish.shimano.com/ja-JP/content/beginners/fishingtackle/lure/index.html)、[Rapala 配色示例](https://www.rapala.com/us_en/blog/countdown-elite-lure-builds-on-rapala-tradition)、[Rapala 拟饵使用示例](https://www.rapala.com/us_en/blog/new-colors-rippin-rap-and-original-floating-lures) 简短整理，**仅是演示知识库，不是经过专家审核的完整教材**。

只索引清单中列出的本目录 Markdown，不索引 README、项目计划书、环境变量或密钥。不下载网页全文，不在运行时联网抓知识。文档内容是回答依据，不是可执行指令。

向量检索默认返回最多 3 个片段，余弦相似度最低阈值为 0.5。该阈值已在当前小型语料的相关/无关示例上检查，但不是回答置信度，也不能保证识别所有知识边界。提示词要求对片段不覆盖的问题明确资料不足，不能拿模型常识冒充检索证据。

来源以 `[K1]` 等标号提供给模型，最终输出附实际检索到的文件名和来源链接。只添加来源并不能自动证明生成的每句话都受支持；涉及产品细节仍需检查片段和具体说明。

新增自有资料的方法：

1. 在 `knowledge_base/` 放入 UTF-8 Markdown。
2. 在 `sources.json` 登记文件名、标题、HTTPS 出处；没有外部链接的自有资料可以设 `url: null`。
3. 重新运行 `index_knowledge.py`。

本阶段只需要 Markdown，使用框架自带文本读取回退；没有额外安装完整 Office/OCR/PDF 解析包。首次索引出现 `MarkItDown not available` 提示是预期的，不影响这些 Markdown 文件。

## 数据隔离与重复索引

商品仍在原集合 `luya_products`，知识使用独立集合 `luya_knowledge_phase3`、命名空间 `luya_phase3`。两者不能配置为同一集合。

分块、来源清单与 embedding 模型共同决定语料版本；点 ID 由版本和分块确定。同一份知识重复索引只覆盖相同点，不重复增加。知识或模型变化后必须重建当前版本的索引，查询不会悄悄使用旧版本。

旧版本点保留，不自动删除集合或商品数据；当前查询只检索当前版本，因此被删除/修改的旧文档不会再参与回答。数据变大后可另行设计旧版本清理，本阶段不做破坏性清理。

## Agent 调用协议与 OpenAI 适配

固定依赖 `hello-agents==0.2.9`，使用该版本的教学型文本协议：模型输出 `[TOOL_CALL:工具名:参数]`，框架解析、执行并把结果送回模型。这是 **LLM 驱动的工具选择，但不是 OpenAI 原生 function calling**。

当前 `gpt-6.1-sol` 保持不变。官方说明它的 Chat Completions 接口不支持原生工具调用，原生工具需使用 Responses API，参见 [official OpenAI documentation：GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol) 和 [Function calling](https://developers.openai.com/api/docs/guides/function-calling)。本阶段使用文本协议，所以不向 Chat Completions 发送原生 `tools` 字段，也不需要迁移已有槽位接口。

`AgentChatLLM` 遵循框架的 `invoke()` 接口，复用本地模型配置，保留 `LLM_EXTRA_BODY` 中的推理参数。Agent 输出为普通文本，不沿用槽位抽取的 JSON 模式，也不发送旧框架默认的温度参数。

每轮最多执行一次库存和一次知识工具，框架工具迭代最多 3 轮；异常参数、未知工具和没有完成的工具标记会被拒绝。库存工具总是用用户原始完整问题抽取条件，防止路由模型重述时遗漏预算。工具只有查询权限，但文本协议仍有模型路由和格式失败风险，不能将本阶段演示视为生产级安全保证。

Phase 1 原有模型槽位抽取仍是独立的一次 LLM 调用。库存问题通常产生路由、槽位抽取、组织回答等多次请求；知识问题通常是路由和回答两次请求。实际费用取决于模型与调用次数，不会因为接入框架变成免费。

## 输出契约与编程调用

```python
from luya_agent import LuyaAgent

app = LuyaAgent()
try:
    result = app.ask("推荐本店100元以内钓鲈鱼的饵，并解释匀收技巧")
    print(result["answer"])
    print([event["tool"] for event in result["tool_calls"]])
finally:
    app.close()
```

返回包含 `question`、`answer`、`tool_calls`、`sources`、`inventory_results` 和 `status`。`tool_calls` 是实际执行结果，不是模型声称调用了工具。

- `ok`：本轮没有工具服务错误；并不表示所有软偏好都满足。
- `partial`：至少一个工具失败，回答保留服务失败提示。
- `no_evidence`：工具均返回空结果，不允许生成器用虚构信息替代空结果。

最终回答附程序生成的库存核验信息和放宽提醒，保留经过工具验证的价格与库存。LLM 的解释仍需要评估，不保证百分之百正确。不要把语义分数当作“适合新手”的可靠概率。

## 验收

```powershell
# Phase 1/2 回归与 Phase 3 契约、工具权限、真实内存 Qdrant 测试
.\.venv\Scripts\python.exe -m unittest discover -s tests

# 本地真实知识检索与集合隔离，不调用模型
.\.venv\Scripts\python.exe verify_phase3.py

# 五条真实模型路由：库存、知识、选色、降级、混合问题
.\.venv\Scripts\python.exe verify_phase3.py --with-llm
```

模拟 LLM 测试只证明工具调用协议和保护逻辑；真实模型测试才检查自主路由。验收不修改商品库存。保留 `verify_phase1.py` 和 `verify_phase2.py` 用于独立回归。

## 文件与使用限制

- `luya_agent.py`：工具适配、注册、统一入口、调用记录和回答证据。
- `agent_llm.py`：兼容现有模型设置的文本 LLM 客户端。
- `knowledge_tool.py`：只读框架 RAG 适配与安全的索引/检索接线。
- `index_knowledge.py`：显式知识索引入口。
- `knowledge_base/`：5 篇演示资料和来源清单。
- `tests/test_phase3.py`、`verify_phase3.py`：契约、真实检索和真实模型验收。

Hello-Agents 0.2.9 的发行元数据标注 CC-BY-NC-SA-4.0（署名、非商业、相同方式共享），参见 [框架项目](https://github.com/jjyaoao/HelloAgents) 和 [对应发行版](https://pypi.org/project/hello-agents/0.2.9/)。此项目按计划书作为学习/作品集演示；若要商业部署，应先核实框架与知识资料授权。没有在本阶段接入外部业务系统。
