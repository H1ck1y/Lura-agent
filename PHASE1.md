# Phase 1：独立库存查询工具

本文记录 Phase 1 严格检索契约。Phase 2 已完成，当前 CLI 默认启用空结果降级，详见 [PHASE2.md](PHASE2.md)。运行本文示例时追加 `--strict` 可复现 Phase 1 行为；编程接口 `InventoryTool.query()`、`search()` 仍保持严格检索，不自动放宽。

流程：用户需求 → LLM 输出 JSON 槽位 → Pydantic 校验 → MariaDB 鱼种别名归一化 → Qdrant 硬过滤 + 向量排序 → 候选商品。

`InventoryTool` 可独立使用。当前支持单一鱼种、人民币单件商品预算上限、水域类型及软性需求。返回结构化商品信息，不生成推荐话术。输入没有指定的硬条件保持 `null`，库存始终要求大于零；空结果直接返回 `status=empty`，不放宽条件。

## 文件

| 文件 | 用途 |
| --- | --- |
| `inventory_tool.py` | `InventoryTool.query()`、`search()` 和命令行交互入口 |
| `llm_slots.py` | 槽位数据结构、few-shot Prompt、Chat Completions 请求和输出校验 |
| `phase1_config.py` | 读取本项目 `.env` 和环境变量 |
| `.env.example` | LLM、MariaDB、Qdrant 配置模板 |
| `examples/*.json` | 不调用 LLM 的结构化查询示例 |
| `tests/test_phase1.py` | 模拟 LLM HTTP 响应 + 真正的 Qdrant 内存过滤测试 |
| `verify_phase1.py` | 本地 MariaDB、Qdrant 和 embedding 模型验收 |

## 第一次运行

PowerShell：

```powershell
cd D:\project\agent\phase0_luya
docker compose up -d
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Phase 0 已同步过商品，不需要重新灌入种子。只有数据库商品更新后，才需要重新运行 `sync_to_qdrant.py`；该脚本的相同商品 ID 会覆盖更新，但不会清理已经从 MariaDB 删除的旧向量。生产环境的增删同步不属于本阶段。

### 先试玩数据层，无需 LLM Key

```powershell
.\.venv\Scripts\python.exe inventory_tool.py --slots-file examples/bass_beginner.json
.\.venv\Scripts\python.exe inventory_tool.py --slots-file examples/species_alias.json --json
```

第二条演示“水老虎”通过数据库别名表归一化为“鳡鱼”。修改示例 JSON，再运行命令即可改变硬条件或软性需求。

### 配置真实 LLM

打开本地 `.env`，填写：

```dotenv
LLM_BASE_URL=你的服务商API基础地址
LLM_MODEL=你的模型名称
LLM_API_KEY=你的API密钥
```

`LLM_BASE_URL` 使用服务商控制台提供的地址，通常包含 `/v1`，不包含末尾的 `/chat/completions`；程序会追加该路径。当前接口使用 Chat Completions 格式，适用于提供该兼容接口的服务。`.env` 已在 `.gitignore` 中忽略。系统环境变量优先于 `.env`。

默认发送 `response_format={"type":"json_object"}`。若你的模型不支持 JSON 模式，可以设 `LLM_JSON_MODE=false`；程序仍会严格校验返回内容，返回无效 JSON 时会停止查询。支持非思考模式的 Qwen 模型可按服务商要求设置 `LLM_EXTRA_BODY={"enable_thinking":false}`，不要给不支持此参数的其他服务添加它。JSON 模式要求与支持范围请参考[百炼结构化输出文档](https://www.alibabacloud.com/help/zh/model-studio/qwen-structured-output)。

配置完成后：

```powershell
.\.venv\Scripts\python.exe inventory_tool.py --query "我是新手，钓鲈鱼，预算一百元以内，最好不容易挂底"
.\.venv\Scripts\python.exe inventory_tool.py --query "钓水老虎，预算两百元以内，想远投" --json
.\.venv\Scripts\python.exe inventory_tool.py --interactive
```

交互模式每次输入独立查询，embedding 模型在进程内复用；输入 `exit` 退出。该终端没有对话记忆。

## 槽位结构与规则

```json
{
  "species": "鲈鱼",
  "max_price": 100,
  "water_type": null,
  "soft_query": "新手友好，容易上手，不易挂底"
}
```

- 四个字段必须全部出现。未指定的硬条件使用 `null`，软需求为空时使用 `""`。
- `max_price` 是有限且非负的数字；`0` 也会执行预算过滤。“预算有限”没有具体数值时不会自动设成 100。
- `water_type` 使用现有数据的 `清水/浑水/咸水/通用`。海水映射为 `咸水`；“淡水”无法区分清浑，保留为软需求，不能假定 `清水`。
- 明确指定鱼种或水域时精确过滤，不自动加入 `通用` 商品。未指定鱼种则不过滤鱼种。
- 多鱼种、预算下限、严格小于和组合总价等无法完整表达的条件会要求用户澄清。
- 未给软需求时用中性的“路亚装备”向量排序。评分是描述的余弦相似度，预算与库存只参与过滤。

编程调用：

```python
from inventory_tool import InventoryTool

tool = InventoryTool()
try:
    result = tool.query("新手钓鲈鱼，预算100元以内，不易挂底")
    print(result["slots"])
    print(result["products"])
finally:
    tool.close()
```

输出包含 `status`、归一化后的 `slots`、`returned_count` 和带 `score` 的 `products`；自然语言入口另包含原始 `query`。`returned_count` 是本次返回的条数，受 `limit` 限制，并非所有匹配商品总数。

## 验证

```powershell
# 不调用真实 LLM：校验异常响应、过滤规则及请求契约
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# 本机真实数据库、向量库与 embedding 模型
.\.venv\Scripts\python.exe verify_phase1.py

# 配置 .env 后：额外调用三次真实 LLM，会产生模型 API 费用
.\.venv\Scripts\python.exe verify_phase1.py --with-llm
```

标准检查依赖 Phase 0 的 31 条种子商品，包括预算 100/40、浑水、黑鱼、别名、未知鱼种、零预算和软需求改变排序。修改商品数据后，预期数量也需调整。

LLM 的语言理解质量必须用真实模型验收。Mock 测试验证请求及处理逻辑，不能证明某个实际模型一定正确提取槽位。
