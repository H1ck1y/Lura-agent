"""Read-only acceptance checks against the project's real database and vectors."""

import argparse
import sys

from inventory_tool import InventoryTool
from llm_slots import Slots
from phase1_config import InventoryError, Settings


def make_slots(species, max_price, water_type=None, soft_query="新手友好，容易上手，不易挂底"):
    return Slots(species=species, max_price=max_price, water_type=water_type, soft_query=soft_query)


def check(result, expected_slots, expected_count=None):
    assert result["slots"] == expected_slots.model_dump(), "槽位不符合预期"
    if expected_count is not None:
        assert result["returned_count"] == expected_count, "商品数量不符合种子数据"
    for product in result["products"]:
        assert product["stock"] > 0, "返回了无库存商品"
        if expected_slots.species is not None:
            assert product["species"] == expected_slots.species, "鱼种不匹配"
        if expected_slots.max_price is not None:
            assert product["price"] <= expected_slots.max_price, "价格超预算"
        if expected_slots.water_type is not None:
            assert product["water_type"] == expected_slots.water_type, "水域不匹配"
    scores = [product["score"] for product in result["products"]]
    assert scores == sorted(scores, reverse=True), "结果没有按语义相似度排序"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Phase 1 真实数据验收")
    parser.add_argument("--with-llm", action="store_true", help="额外调用真实 LLM，需要本地配置；会产生 API 费用")
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.with_llm:
        settings.require_llm()
    tool = InventoryTool(settings)
    try:
        cases = [
            ("鲈鱼预算100", make_slots("鲈鱼", 100), make_slots("鲈鱼", 100), 6),
            ("鲈鱼预算40", make_slots("鲈鱼", 40), make_slots("鲈鱼", 40), 2),
            ("鲈鱼浑水", make_slots("鲈鱼", 100, "浑水"), make_slots("鲈鱼", 100, "浑水"), 3),
            ("黑鱼预算50", make_slots("黑鱼", 50), make_slots("黑鱼", 50), 1),
            ("俗称水老虎", make_slots("水老虎", 200, "清水"), make_slots("鳡鱼", 200, "清水"), 1),
            ("未知鱼种空结果", make_slots("测试不存在的鱼种", 100), make_slots("测试不存在的鱼种", 100), 0),
            ("零预算不放宽", make_slots("鲈鱼", 0), make_slots("鲈鱼", 0), 0),
        ]
        for name, inputs, expected, count in cases:
            result = tool.search(inputs)
            check(result, expected, count)
            print(f"PASS {name}: {result['returned_count']} 件商品")
        beginner = tool.search(make_slots("鲈鱼", 100))
        expert = tool.search(make_slots("鲈鱼", 100, soft_query="重心靠后，抛投距离远，复杂障碍区精准打点，适合有经验的钓手"))
        assert beginner["products"][0]["id"] != expert["products"][0]["id"], "软需求变化没有影响排名"
        print("PASS 语义排序变化: " + beginner["products"][0]["name"] + " -> " + expert["products"][0]["name"])
        if args.with_llm:
            for query, species, budget, water, count in [
                ("我是新手，钓鲈鱼，预算一百元以内，最好不容易挂底", "鲈鱼", 100, None, 6),
                ("钓水老虎，预算两百元以内，远投", "鳡鱼", 200, None, 1),
                ("在浑水钓黑鱼，预算50元以内", "黑鱼", 50, "浑水", 1),
            ]:
                result = tool.query(query)
                actual = Slots.model_validate(result["slots"])
                assert (actual.species, actual.max_price, actual.water_type) == (species, budget, water)
                check(result, actual, count)
                print("PASS 真实 LLM: " + query)
        else:
            print("真实 LLM 验证未运行；配置 .env 后加 --with-llm 执行。")
    finally:
        tool.close()


if __name__ == "__main__":
    try:
        main()
    except (InventoryError, AssertionError) as exc:
        print("FAIL: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
