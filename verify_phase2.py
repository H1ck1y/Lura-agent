"""Read-only Phase 2 acceptance over the existing 31 seeded products."""

import argparse
import sys

from inventory_tool import InventoryTool
from llm_slots import Slots
from phase1_config import InventoryError, Settings


def slots(species, budget, water=None):
    return Slots(species=species, max_price=budget, water_type=water, soft_query="新手友好，不易挂底")


def check(result, original, status, stages):
    assert result["status"] == status, "状态不符"
    assert [a["stage"] for a in result["attempts"]] == stages, "重试顺序不符"
    assert result["slots"] == original.model_dump(), "原始条件被修改"
    assert result["reply"], "缺少兜底话术"
    assert len(result["attempts"]) <= 3, "重试失控"
    if status == "empty":
        assert not result["products"] and result["returned_count"] == 0
    else:
        assert result["products"], "缺少候选"
    final = result["attempts"][-1]
    for product in result["products"]:
        assert product["stock"] > 0, "返回缺货商品"
        assert product["species"] == original.species, "偷换了目标鱼种"
        if final["water_types"] is not None:
            assert product["water_type"] in final["water_types"], "水域越界"
        if original.max_price is not None:
            assert product["price"] <= final["slots"]["max_price"], "预算扩张越界"
        expected = []
        if original.water_type is not None and product["water_type"] != original.water_type:
            expected.append("water_type")
        if original.max_price is not None and product["price"] > original.max_price:
            expected.append("max_price")
        assert product["relaxed_fields"] == expected, "没有正确标注条件差异"
        assert product["matches_original_filters"] == (not expected)
    scores = [p["score"] for p in result["products"]]
    assert scores == sorted(scores, reverse=True), "排序不符"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Phase 2 空结果降级验收")
    parser.add_argument("--with-llm", action="store_true", help="额外测试三条真实 LLM 查询，会产生 API 费用")
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.with_llm:
        settings.require_llm()
    tool = InventoryTool(settings)
    try:
        cases = [
            ("精确命中不重试", slots("鲈鱼", 100), "ok", ["strict"]),
            ("通用水域替代", slots("鳡鱼", 600, "浑水"), "alternatives", ["strict", "generic_water"]),
            ("预算上浮命中", slots("黑鱼", 40, "浑水"), "alternatives", ["strict", "generic_water", "budget"]),
            ("水域与预算共同放宽", slots("鳡鱼", 500, "咸水"), "alternatives", ["strict", "generic_water", "budget"]),
            ("不混用清水与咸水", slots("青甘", 1000, "清水"), "empty", ["strict", "generic_water", "budget"]),
            ("未知鱼种兜底", slots("不存在鱼种", 100), "empty", ["strict", "budget"]),
            ("零预算不扩张", slots("鲈鱼", 0), "empty", ["strict"]),
        ]
        for name, original, status, stages in cases:
            result = tool.search_with_fallback(original)
            check(result, original, status, stages)
            print(f"PASS {name}: {status}, {len(result['attempts'])} 次, {result['returned_count']} 件")
        alias = tool.search_with_fallback(slots("水老虎", 500, "浑水"))
        check(alias, slots("鳡鱼", 500, "浑水"), "alternatives", ["strict", "generic_water", "budget"])
        print("PASS 别名归一化后降级")
        disabled = tool.search_with_fallback(slots("黑鱼", 40, "浑水"), allow_budget_relaxation=False)
        check(disabled, slots("黑鱼", 40, "浑水"), "empty", ["strict", "generic_water"])
        print("PASS 关闭预算放宽")
        if args.with_llm:
            for query, original, status, stages in [
                ("新手在浑水钓黑鱼，预算40元以内，不易挂底", slots("黑鱼", 40, "浑水"), "alternatives", ["strict", "generic_water", "budget"]),
                ("在浑水钓水老虎，预算500元以内，想远投", slots("鳡鱼", 500, "浑水"), "alternatives", ["strict", "generic_water", "budget"]),
                ("在清水钓青甘，预算1000元以内", slots("青甘", 1000, "清水"), "empty", ["strict", "generic_water", "budget"]),
            ]:
                result = tool.query_with_fallback(query)
                actual = Slots.model_validate(result["slots"])
                assert (actual.species, actual.max_price, actual.water_type) == (original.species, original.max_price, original.water_type)
                check(result, actual, status, stages)
                print("PASS 真实 LLM: " + query)
                print("  " + result["reply"])
        else:
            print("真实 LLM 未运行；加 --with-llm 可测试自然语言到降级结果的完整链路。")
    finally:
        tool.close()


if __name__ == "__main__":
    try:
        main()
    except (InventoryError, AssertionError) as exc:
        print("FAIL: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
