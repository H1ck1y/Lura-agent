# Phase 3.1：混合检索、重排与可选查询扩展

后续 [Phase 4](PHASE4.md) 已为当前命令行加入简化记忆；本文“没有记忆”描述的是 Phase 3.1 阶段范围，`--no-memory` 可复现。检索模式和扩展开关不变。

这是 Phase 3 的知识检索增强，不是 Phase 4。默认 **BM25＋向量混合召回＋本地 CrossEncoder 重排**；MQE、HyDE 默认关闭。没有添加 GraphRAG、Hydra 配置框架、对话记忆或前端。库存工具与预算放宽规则未改变。

## 直接试玩

在 PowerShell 中：

```powershell
cd D:\project\agent\phase0_luya
docker compose up -d
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 有 Phase 3 的现成知识索引且资料没变，无需重跑；新环境先运行：
.\.venv\Scripts\python.exe index_knowledge.py

# 默认：混合召回 + 重排
.\.venv\Scripts\python.exe luya_agent.py --interactive --trace
```

问 `铅头钩配软饵为什么容易挂底？`、`水里泥很多看不清，拟饵选什么颜色？`。`--trace` 显示实际工具、检索模式、是否执行重排、额外 LLM 调用数、检索耗时和扩展失败提示；不显示模型思考或假设答案。

不同模式可以这样对比：

```powershell
# 原 Phase 3 基础向量检索，不重排、不扩展
.\.venv\Scripts\python.exe luya_agent.py --query "铅头钩为什么容易挂底？" --rag-mode basic --trace

# 只混合召回，不重排，用于消融
.\.venv\Scripts\python.exe luya_agent.py --query "铅头钩为什么容易挂底？" --no-rerank --trace

# 可选扩展：会调用本地 .env 中配置的 OpenAI 模型，产生额外费用
.\.venv\Scripts\python.exe luya_agent.py --query "铅头钩为什么容易挂底？" --mqe --trace
.\.venv\Scripts\python.exe luya_agent.py --query "铅头钩为什么容易挂底？" --hyde --trace
.\.venv\Scripts\python.exe luya_agent.py --query "铅头钩为什么容易挂底？" --mqe --hyde --json
```

不改 `.env` 也能使用默认配置。环境变量见 `.env.example`：`RAG_MODE=hybrid`、`RAG_RERANK=true`、`RAG_MQE=false`、`RAG_HYDE=false`、`RAG_CANDIDATE_K=12`、`RAG_MIN_RERANK_SCORE=0.1`。命令行覆盖环境变量；`--no-mqe`/`--no-hyde` 可显式关闭已配置的开关。`basic` 模式不允许同时启用扩展。

首次知识重排需下载 `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`。权重约 470 MB，默认 CPU，后续使用本地 Hugging Face 缓存；不会请求收费的重排服务。首次加载耗时与热查询不同。模型下载失败会明确报错，可主动选择 `--no-rerank` 或 `--rag-mode basic`，不会偷偷把普通排序宣称为重排。

## 增强在哪里

| 模式 | 处理过程 | 额外生成模型调用 |
| --- | --- | --- |
| basic | 原问题 → BGE → Qdrant cosine Top 3 | 0 |
| hybrid，无重排 | 原问题 → BGE + 中文 BM25 → RRF 融合 → Top 3 | 0 |
| 默认 hybrid，重排 | 混合候选 → CrossEncoder 对“原问题、片段”逐对评分 → Top 3 | 0 |
| 加 MQE | 最多两条改写查询 → 分别检索 → 合并真实索引候选 → 按原问题重排 | 最多 1 |
| 加 HyDE | 生成假设说明 → embedding 检索 → 合并真实索引候选 → 按原问题重排 | 最多 1 |

这里的“0”仅指**知识检索额外调用**。Agent 路由和组织最终回答仍使用 LLM，库存槽位抽取也仍独立收费；不能理解为整个 Agent 免费。

- BM25：`jieba` 中文搜索分词，加入少量路亚词；`rank-bm25` 的 BM25Plus，`delta=0`，使小语料 IDF 为正且无匹配词的片段得零分。它补充向量可能漏掉的明确术语。
- 向量：继续使用已缓存的 `BAAI/bge-small-zh-v1.5` 和现有 Qdrant 知识集合。
- RRF：按排名累计 `1 / (60 + rank)`，按分块 ID 去重，不把 BM25 原始分数和 cosine 直接相加。
- Reranker：通过 `sentence-transformers.CrossEncoder` 将完整问题与候选片段联合编码，区别于独立计算 query/document embedding。默认模型来自 [Sentence Transformers 模型卡](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)，并非“路亚专用训练模型”。
- MQE/HyDE：项目自研的有界生成/校验逻辑，并非直接启用框架默认云管线。[HyDE 原论文](https://arxiv.org/abs/2212.10496) 中的假设文档可能包含虚构内容，因此这里绝不把它作为来源或答案证据。

“基础设施调包，业务流程自己实现”的边界：模型与 BM25 数学实现复用依赖；版本过滤、中文词处理、混合融合、原问题安全门、扩展开关、失败回退、来源输出和评估由本项目实现。没有改 `.venv` 中的框架代码。

## 分数、来源与边界

结构化结果中每个片段包含：

- `score` / `score_type`：basic 为 `cosine`；无重排混合为 `rrf`；重排为 `reranker_sigmoid`。
- `scores`：保留各检索通道原始分数；执行重排时也保留 RRF 分数。
- 原有 `citation`、`source_file`、标题、链接、内容：始终来自当前索引与来源清单。

重排分数为模型 logit 经 sigmoid 的结果，**不是校准后的准确率、可信度或答案正确概率**。默认以 0.1 过滤明显不相关候选；阈值可配置，需在扩展语料和独立题目上重评。

查询只使用当前命名空间、当前语料版本、同一 embedding 模型的索引。BM25 从这些真实索引分块构建，不直接把目录下所有文件读成证据；本地缓存随版本/分块数量变化或索引重建失效。不改商品集合，不删除旧知识版本。

扩展采用保守策略：先做原问题召回，原问题完全无候选则不调用 MQE/HyDE。扩展可以召回新的真实索引分块，但须由 Reranker 对**原问题**重新评分并通过阈值；不会用假设说明代替原问题进行最终相关性判断。显式禁用重排时，扩展只参与原问题已有候选的融合排序。这限制了“救回完全漏检问题”的能力，但避免纯假设文本成为无关问题的唯一检索依据。它们属于可选实验，不承诺一定改善当前演示语料，也不是完全防幻觉的保证。

每个开关最多一次生成调用、无自动重试；生成 JSON 需本地校验，失败会保留原问题检索并给出告警。沿用现有模型/地址/推理参数。按照 [OpenAI Docs 的 JSON 模式说明](https://developers.openai.com/api/docs/guides/structured-outputs)，JSON 格式不等于符合业务 schema，因此还检查字段、条数、长度和类型。只在扩展请求中启用 JSON 模式，Agent 正常回答仍为文本。

Qdrant/embedding/重排故障是服务错误，不伪装成“没有知识”；只有真正无候选或候选相关性过低才返回资料不足。全部片段引用仍不能自动证明最终回答每句话都正确。

## 检查效果、耗时和调用成本

```powershell
# 不收费：对同一份 12 题比较 basic / hybrid / rerank
.\.venv\Scripts\python.exe evaluate_rag.py

# 收费：同样的前两题，对照三个基础模式及 MQE / HyDE / both
.\.venv\Scripts\python.exe evaluate_rag.py --with-llm --limit 2

# 只验证组合扩展，最多两次额外生成请求
.\.venv\Scripts\python.exe evaluate_rag.py --with-llm --variants both --limit 1

.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe verify_phase3.py --with-llm
```

题目在 `evals/rag_cases.json`：8 个相关问题及改写、4 个无关/不覆盖问题。标注预期文件，输出 Hit@1、Recall@3、MRR@3、无关问题拒答率、查询耗时中位数和额外 API 请求/token 数。先进行无生成模型调用的预热，预热耗时单独记录，不计入查询耗时。扩展失败告警也计数；`usage_complete=false` 时 token 总数不完整，不能按零费用解释。

这只是与当前 5 篇演示资料配套的检索 smoke test，**不是独立测试集、生产 benchmark 或最终答案质量评测**。命中率高也不表示知识已完整。美元费用不作猜测，报告提供 API 实际 token 用量，不包含本地计算、模型下载成本或路由/最终回答费用。

本地 BM25 演示实现最多 5000 分块，超出会提示迁移服务端稀疏索引；不把内存演示索引当成生产规模方案。下一步应扩展经过核验的知识资料与独立评测集，再决定哪些实验开关值得默认启用。

### 本机验收记录（2026-10-03）

依赖已安装、重排模型已下载。全部 89 项单元/契约测试通过，`pip check` 无依赖冲突；真实 Qdrant 6 项检查及 5 类 OpenAI Agent 路由场景通过。验证包含纯库存、纯知识、预算放宽、混合问句和真实 MQE＋HyDE 知识回答；商品集合仍是 31 条。

同一份 12 题、共享模型并排除预热后的示例结果：

| 模式 | 相关题 Hit@1 / Recall@3 | 4 道不覆盖题拒答率 | 检索耗时中位数 | 相关题平均片段数 |
| --- | --- | --- | --- | --- |
| 基础向量 | 100% / 100% | 100% | 19.39ms | 2.5 |
| 混合，无重排 | 100% / 100% | 50% | 18.04ms | 2.875 |
| 默认混合＋重排 | 100% / 100% | 100% | 102.53ms | 1 |

**不能宣称召回提升**：基础模式在这组小题上本来已全部命中。裸 BM25 混合增加了词面误召回，默认重排将本例的误召回过滤掉，且减少输出片段，但付出更多 CPU 耗时。几十毫秒的差值不能当作稳定速度结论。

MQE、HyDE、组合开关各在同一道挂底问题上完成真实生成检索：额外调用分别为 1、1、2 次，告警为 0，均命中预期文件。该次预热后检索约 6.5s、9.7s、17.2s；API 返回的额外输入/输出 token 分别为 98/43、97/190、195/226。这只是一次功能检查，不是质量收益证明，也不保证下次相同耗时或用量。最终组合 Agent 的真实来源仍是知识文件，而非假设文本。因此两个实验开关仍保持默认关闭。

真实验收也发现并修正了混合问句的输入问题：库存工具继续绑定用户完整需求；路由提示明确要求知识工具仅提取相关知识子问题，避免把商品预算/购买条件当成文档必须覆盖的内容。提示词本身仍可能失效，不能视为绝对正确的解析器。

## 文件与编程调用

`rag_retrieval.py`：配置、中文 BM25、RRF、懒加载重排器、有界扩展；`knowledge_tool.py`：现有框架知识工具集成；`evaluate_rag.py`：只读对比；`tests/test_phase3_1.py`：新增契约测试。

```python
from luya_agent import LuyaAgent
from rag_retrieval import RetrievalOptions

app = LuyaAgent(rag_options=RetrievalOptions(mqe=True, hyde=False))
try:
    result = app.ask("铅头钩为什么容易挂底？")
    print(result["answer"])
finally:
    app.close()
```

Hello-Agents 0.2.9 的非商业许可证限制仍需注意，详见 Phase 3；新增检索技术不会改变该框架授权。
