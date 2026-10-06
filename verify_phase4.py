"""Real Phase 4 multi-turn and opt-in persistence acceptance (explicit API flag)."""

import argparse
from pathlib import Path
import sys
import tempfile

from luya_memory import ConversationMemory, PreferenceStore
from memory_agent import MemoryLuyaAgent
from phase1_config import InventoryError, Settings
from rag_retrieval import RetrievalOptions


def check_inventory(app, question, species, budget):
    result = app.ask(question)
    assert result["status"] in {"ok", "no_evidence"}, "模型或工具失败"
    assert result["inventory_results"], "未查询当前库存"
    inventory = result["inventory_results"][-1]
    assert inventory["slots"]["species"] == species, "鱼种继承/覆盖失败"
    assert inventory["slots"]["max_price"] == budget, "预算继承/覆盖失败"
    assert all(p["stock"] > 0 for p in inventory["products"])
    assert all(p["species"] == species for p in inventory["products"])
    assert all(p["price"] <= budget for p in inventory["products"]) if budget is not None else True
    print(f"PASS {question} → 鱼种={species}，预算={budget}，候选={len(inventory['products'])}")
    return result


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Phase 4 真实验收；临时独立偏好文件，不改真实用户记忆")
    parser.add_argument("--with-llm", action="store_true", help="允许真实 API 多轮测试，会产生费用")
    args = parser.parse_args()
    if not args.with_llm:
        parser.error("请显式使用 --with-llm；不收费的测试用 unittest。")
    settings = Settings.from_env()
    settings.require_llm()
    with tempfile.TemporaryDirectory(prefix="luya-phase4-verify-") as directory:
        path = Path(directory) / "memory.sqlite3"
        app = MemoryLuyaAgent(settings, user_id="alice", memory_path=path,
                              allow_budget_relaxation=False, rag_options=RetrievalOptions())
        try:
            initial_products = app.rag.client.count(settings.collection, exact=True).count
            check_inventory(app, "我是新手，钓鲈鱼，预算100元以内，推荐本店商品", "鲈鱼", 100)
            check_inventory(app, "再便宜一点，继续推荐本店商品", "鲈鱼", 100)
            check_inventory(app, "预算改成50元以内，继续推荐", "鲈鱼", 50)
            check_inventory(app, "换成钓黑鱼，继续推荐本店商品", "黑鱼", 50)
            check_inventory(app, "这次不限制预算，继续推荐", "黑鱼", None)
            app.ask('/remember {"species":"鲈鱼","max_price":100,"soft_query":"新手友好，不易挂底"}')
            app.ask("/new")
            check_inventory(app, "按我的偏好推荐本店商品", "鲈鱼", 100)
            result = app.ask("铅头钩搭配软饵为什么容易挂底？只解释原理")
            assert result["sources"], "知识问题未检索到来源"
            assert "rag" in {event["tool"] for event in result["tool_calls"]}
            print("PASS 知识路径写回短期上下文")
            result = app.ask("新手怎样练习匀速收线？只说钓法，不买商品")
            assert result["sources"], "匀收知识未检索到来源"
            result = app.ask("那轻抽呢？只讲钓法")
            assert "rag" in {event["tool"] for event in result["tool_calls"]}
            assert any(s["source_file"] == "03_retrieve.md" for s in result["sources"]), "知识追问未命中真实资料"
            print("PASS 知识追问的上下文与本轮来源")
            assert app.rag.client.count(settings.collection, exact=True).count == initial_products == 31
        finally:
            app.close()
        reopened = MemoryLuyaAgent(settings, user_id="alice", memory_path=path,
                                  allow_budget_relaxation=False, rag_options=RetrievalOptions())
        try:
            check_inventory(reopened, "按我保存的偏好推荐本店商品", "鲈鱼", 100)
            assert not reopened.memory.store.load("bob"), "长期偏好跨用户泄漏"
            reopened.ask("/forget")
            assert reopened.memory.snapshot()["defaults"] == {}, "删除偏好后仍有残留"
            print("PASS 重启持久化、用户隔离、遗忘，以及商品数据不变")
        finally:
            reopened.close()


if __name__ == "__main__":
    try:
        main()
    except (InventoryError, AssertionError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
