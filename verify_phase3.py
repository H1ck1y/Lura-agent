"""Phase 3 retrieval checks and optional real-model autonomous routing acceptance."""

import argparse
import json
import sys

from knowledge_tool import KnowledgeRAGTool
from luya_agent import LuyaAgent
from phase1_config import InventoryError, Settings


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Phase 3 RAG 与 Agent 真实验收")
    parser.add_argument("--with-llm", action="store_true", help="调用真实 LLM 检查工具路由与回答，会产生 API 费用")
    args = parser.parse_args()
    settings = Settings.from_env()
    rag = KnowledgeRAGTool(settings)
    try:
        for query, expected_file in [
            ("铅头钩配软饵为什么容易挂底？", "02_jighead_snags.md"),
            ("清水和浑水怎么选拟饵颜色？", "04_water_color.md"),
            ("新手怎样练习匀速收线？", "03_retrieve.md"),
            ("米诺怎么操作？", "05_stop_go.md"),
        ]:
            result = rag.search_result(query)
            assert result["status"] == "ok", "相关问题没有召回资料"
            assert expected_file in {s["source_file"] for s in result["snippets"]}, "没有召回预期文档"
            assert all(s["url"] for s in result["snippets"]), "缺少来源"
            print("PASS 真实 RAG: " + query)
        result = rag.search_result("量子计算机怎么纠错？")
        assert result["status"] == "empty", "无关问题没有拒绝"
        print("PASS 资料不足兜底")
        products_before = rag.client.count(settings.collection, exact=True).count
        assert products_before == 31, "现有种子商品数量变化"
        print("PASS 商品与知识集合隔离：商品仍为 31 条")
    finally:
        rag.close()
    if not args.with_llm:
        print("真实 Agent 模型路由未运行；加 --with-llm 验证完整链路。")
        return
    settings.require_llm()
    app = LuyaAgent(settings)
    try:
        for question, expected_tools in [
            ("我是新手，钓鲈鱼，预算100元以内，最好不容易挂底，推荐本店商品", {"inventory"}),
            ("铅头钩搭配软饵为什么容易挂底？只讲原理，不买东西", {"rag"}),
            ("清水和浑水怎么选择拟饵颜色？", {"rag"}),
            ("在浑水钓黑鱼，预算40元以内，请推荐本店商品", {"inventory"}),
            ("推荐本店100元以内钓鲈鱼的饵，并解释新手怎么练习匀速收线", {"inventory", "rag"}),
        ]:
            result = app.ask(question)
            actual_tools = {e["tool"] for e in result["tool_calls"]}
            assert actual_tools == expected_tools, f"路由不符：{actual_tools}"
            assert result["status"] == "ok", "工具执行失败"
            assert result["answer"].strip() and "[TOOL_CALL:" not in result["answer"]
            assert app.agent._history == [], "意外保留会话记忆"
            if "rag" in expected_tools:
                assert result["sources"], "知识回答缺少来源"
                assert any(f"[{s['citation']}]" in result["answer"] for s in result["sources"])
            if "inventory" in expected_tools:
                assert result["inventory_results"], "库存推荐没有证据"
                for inventory in result["inventory_results"]:
                    assert inventory["products"], "库存没有候选"
                    assert all(p["stock"] > 0 for p in inventory["products"])
                    assert any(p["name"] in result["answer"] for p in inventory["products"])
                    if inventory["status"] == "alternatives":
                        assert "超过原预算" in result["answer"] and "确认" in result["answer"]
            print("PASS 真实 Agent: " + question)
            print("  工具：" + " → ".join(e["tool"] for e in result["tool_calls"]))
            print("  回答：" + result["answer"][:500].replace("\n", " "))
        assert app.rag.client.count(settings.collection, exact=True).count == products_before, "库存被修改"
    finally:
        app.close()


if __name__ == "__main__":
    try:
        main()
    except (InventoryError, AssertionError) as exc:
        print("FAIL: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
