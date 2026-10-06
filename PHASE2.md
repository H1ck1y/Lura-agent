# Phase 2：空结果降级重试与兜底回复

对应项目计划书的阶段 2：候选为空时放宽条件重试，生成合理兜底话术。仅扩展 `InventoryTool`；没有接入 ToolRegistry、RAG、对话记忆或前端，也没有生成完整的客服推荐话术。

## 降级规则

计划书没有指定具体放宽顺序。此版本采用有边界、可追踪的确定性策略：

1. 按原始条件严格查一次。有商品立即返回，不为了更多结果而重试。
2. 如果用户指定了清水、浑水或咸水，额外允许数据库标注为“通用”水域的商品，再查一次。**不把清水、浑水、咸水互相替换**。“通用”只是当前商品标签，不保证真实场景适用，输出会提示用户确认。
3. 仍无候选且给出了正数预算，预算上限最多上浮 20%，向下取整到分，再查一次。保留上一步的水域范围。这些是需要用户确认的超预算替代品，**不是满足原预算的推荐**。
4. 仍为空：返回 `status=empty`、尝试记录和确认鱼种/调整条件的兜底回复，不编造商品。

最多 3 次 Qdrant 查询，不适用的步骤直接跳过。未指定预算不编造预算；0 元不扩张；已经是“通用”水域不重复扩张。预算计算无法得到更高的有限数值时也不重试。

始终要求 `stock > 0`，不替换或删除指定的鱼种。俗称先通过 MariaDB 别名表归一化；每轮保留相同软性需求和原来的向量评分方式。LLM 只抽取一次槽位，重试与回复由程序处理，不额外调用模型，因此重试不会额外产生模型调用费用。

兜底回复使用模板，根据真实候选和条件差异生成，避免模型编造价格、库存或放宽原因。无结果不报异常；模型、数据库、向量库故障仍作为服务错误报告，不伪装成“没有库存”。交互终端捕获这些错误后允许继续输入。

## 手动试玩

PowerShell：

```powershell
cd D:\project\agent\phase0_luya
docker compose up -d
.\.venv\Scripts\python.exe inventory_tool.py --interactive
```

现有 `.env` 的 LLM 配置直接复用，不需要新增依赖。首次加载 embedding 模型可能较慢；同一交互进程会复用模型。输入 `exit` 退出，每次输入独立，没有上下文记忆。

建议输入：

| 输入 | 现有种子数据下预期 |
| --- | --- |
| `钓鲈鱼，预算100元以内，新手友好` | 精确命中，不重试 |
| `在浑水钓鳡鱼，预算600元以内` | “通用”水域替代候选，提示确认适用性 |
| `新手在浑水钓黑鱼，预算40元以内，不易挂底` | 放宽到 48 元，雷蛙水面系超原预算 8 元；软偏好并不保证满足 |
| `在浑水钓水老虎，预算500元以内，想远投` | 归一化为鳡鱼，允许“通用”水域，预算上限 600 元 |
| `在清水钓青甘，预算1000元以内` | 清水不能替换成咸水，最终给出空结果兜底 |
| `钓鲈鱼，预算0元以内` | 无候选，不扩张零预算 |

单条查询与完整输出：

```powershell
.\.venv\Scripts\python.exe inventory_tool.py --query "在浑水钓黑鱼，预算40元以内" --json
```

如果预算不能增加：

```powershell
.\.venv\Scripts\python.exe inventory_tool.py --interactive --no-budget-relaxation
```

如果任何硬条件都不能放宽，使用 Phase 1 严格模式：

```powershell
.\.venv\Scripts\python.exe inventory_tool.py --interactive --strict
```

`--strict` 优先，不执行任何降级。当前槽位结构没有“禁止超预算”字段；输入中的预算会被保存为原上限，但默认仍可能展示明确标注的超预算替代品。需要禁止展示这类替代品时，使用以上开关，不依赖一句“绝对不能超预算”的措辞。

不调用 LLM 的试玩方式：

```powershell
.\.venv\Scripts\python.exe inventory_tool.py --slots-file examples/blackfish_low_budget.json
.\.venv\Scripts\python.exe inventory_tool.py --slots-file examples/blackfish_low_budget.json --strict
```

## 返回契约与编程调用

保留 Phase 1 的 `query()` 和 `search()` 严格接口；增加显式 Phase 2 接口，CLI 默认调用后者：

```python
from inventory_tool import InventoryTool

tool = InventoryTool()
try:
    result = tool.query_with_fallback("在浑水钓黑鱼，预算40元以内")
    print(result["reply"])
    print(result["attempts"])
    print(result["products"])
finally:
    tool.close()
```

结构化入口是 `search_with_fallback(slots, limit=10, allow_budget_relaxation=True)`；自然语言入口也支持相同参数。

| 字段 | 含义 |
| --- | --- |
| `status` | `ok` 精确命中；`alternatives` 放宽后找到替代候选；`empty` 最终无候选 |
| `slots` | 别名归一化后的原始条件，预算保持原值 |
| `effective_slots` | 最后一轮使用的槽位，预算可能变化；水域的 OR 范围见 `attempts[-1].water_types` |
| `attempts` | 每轮阶段、槽位、允许水域和本次返回数量 |
| `relaxations` | 执行过的放宽项，包含原条件与允许值；即使最终为空也保留 |
| `reply` | 精确命中/替代候选/最终空结果的中文回复 |
| `products[*].matches_original_filters` | 该商品是否符合原始结构化条件，不代表满足所有软偏好 |
| `products[*].relaxed_fields` | 该商品偏离的原条件字段，如 `max_price`、`water_type` |
| `products[*].deviations` | 可展示给用户的具体差异，如超预算金额 |

商品仍包含原始价格、库存、描述和 `score`。`score` 仍是描述向量的余弦相似度，不是 LLM 推荐评分。`returned_count` 受 `limit` 限制，不是匹配总量。任何替代候选都不自动表示用户已接受放宽。

## 测试与验收

```powershell
# Phase 1 回归 + Phase 2 边界测试，使用真实内存 Qdrant
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# 真实 MariaDB / Qdrant / embedding 验收，无模型 API 费用
.\.venv\Scripts\python.exe verify_phase2.py

# 额外三条真实模型全链路测试，会产生 API 费用
.\.venv\Scripts\python.exe verify_phase2.py --with-llm
```

测试覆盖重试顺序、命中即停止、预算边界、别名、零预算、未指定条件、未知鱼种、缺货过滤、清咸水隔离、排序、参数开关、有限次数、原输入不变、模型仅抽取一次以及服务失败不当成空库存。

真实数据验收依赖 Phase 0 的 31 条种子商品，脚本只读，不改库存、不重灌数据库。商品数据修改后，相关验收预期也应更新。

OpenAI 调用沿用 Phase 1 的 JSON 模式和本地槽位校验。JSON 模式只保证 JSON 格式，不保证字段符合应用结构，仍须校验；参见 [official OpenAI documentation：Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)。本阶段不迁移模型或接口。

## 新增文件

- `fallback_logic.py`：有界重试、预算计算、差异标注与回复模板。
- `tests/test_phase2.py`：19 项 Phase 2 单元测试。
- `verify_phase2.py`：9 项真实数据检查，可追加 3 项真实模型查询。
- `examples/blackfish_low_budget.json`：低预算空结果演示。

`inventory_tool.py` 新增 Phase 2 接口与命令行开关，Phase 1 严格接口保持兼容。
