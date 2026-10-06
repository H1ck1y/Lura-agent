"""Bounded, deterministic Phase 2 relaxation; never changes species or stock."""

import math
from decimal import Decimal, ROUND_DOWN, localcontext

from llm_slots import Slots


def expanded_budget(price):
    """At most +20%, rounded down to cents; zero/overflow cannot be relaxed."""
    if price is None or price <= 0:
        return None
    with localcontext() as context:
        context.prec = 400
        value = float((Decimal(str(price)) * Decimal("1.2")).quantize(
            Decimal("0.01"), rounding=ROUND_DOWN
        ))
    return value if math.isfinite(value) and value > price else None


def budget_text(value):
    return format(value, ".2f").rstrip("0").rstrip(".")


def make_reply(result, original):
    if result["status"] == "ok":
        return f"找到 {result['returned_count']} 件符合原鱼种、预算和水域筛选条件的有库存商品；软性需求仅用于排序。"
    if result["status"] == "empty":
        name = f"“{original.species}”" if original.species else "你的需求"
        retry_note = "，有限放宽后也没有候选" if len(result["attempts"]) > 1 else ""
        return (
            f"当前库存未找到符合{name}查询条件的商品{retry_note}。"
            "可以确认鱼种名称、调整预算或补充作钓水域再查；"
            "未改换目标鱼种，也未推荐缺货商品。"
        )
    notices = []
    if any("water_type" in p["relaxed_fields"] for p in result["products"]):
        notices.append(f"部分商品水域标注为“通用”，并非精确标注“{original.water_type}”，需确认实际适用性")
    if any("max_price" in p["relaxed_fields"] for p in result["products"]):
        notices.append(
            f"部分商品超过原预算 {budget_text(original.max_price)} 元，"
            f"本轮上限为 {budget_text(result['effective_slots']['max_price'])} 元（最多上浮 20%）"
        )
    return (
        f"原条件没有匹配商品，找到 {result['returned_count']} 件有库存的替代候选。"
        + "；".join(notices) + "。这些不是完全满足原条件的推荐，需要你确认是否接受。"
    )


def search_with_fallback(search, original: Slots, limit, *, allow_budget_relaxation=True):
    effective = original
    include_generic = False
    stage = "strict"
    relaxations = []
    attempts = []
    budget = expanded_budget(original.max_price) if allow_budget_relaxation else None
    # Strict -> original water OR generic -> budget +20%. At most three queries.
    while True:
        result = search(effective, limit, include_generic_water=include_generic)
        water_types = None if effective.water_type is None else [effective.water_type]
        if include_generic:
            water_types.append("通用")
        attempts.append({
            "stage": stage, "slots": effective.model_dump(),
            "water_types": water_types, "returned_count": result["returned_count"],
        })
        if result["products"]:
            break
        if not include_generic and original.water_type not in {None, "通用"}:
            include_generic = True
            stage = "generic_water"
            relaxations.append({
                "field": "water_type", "requested": original.water_type,
                "allowed": [original.water_type, "通用"],
            })
            continue
        if budget is not None and effective.max_price == original.max_price:
            effective = original.model_copy(update={"max_price": budget})
            stage = "budget"
            relaxations.append({
                "field": "max_price", "requested": original.max_price, "allowed": budget,
            })
            continue
        break

    products = []
    for product in result["products"]:
        fields = []
        deviations = []
        if original.water_type is not None and product["water_type"] != original.water_type:
            fields.append("water_type")
            deviations.append(f"水域标注为{product['water_type']}，原要求为{original.water_type}")
        if original.max_price is not None and product["price"] > original.max_price:
            fields.append("max_price")
            difference = Decimal(str(product["price"])) - Decimal(str(original.max_price))
            deviations.append(f"超过原预算 {budget_text(float(difference))} 元")
        products.append({
            **product, "matches_original_filters": not fields,
            "relaxed_fields": fields, "deviations": deviations,
        })
    output = {
        **result, "slots": original.model_dump(), "effective_slots": effective.model_dump(),
        "status": "empty" if not products else "ok" if len(attempts) == 1 else "alternatives",
        "products": products, "attempts": attempts, "relaxations": relaxations,
    }
    output["reply"] = make_reply(output, original)
    return output
