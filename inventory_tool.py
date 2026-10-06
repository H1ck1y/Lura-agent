"""Phase 1: natural-language slots, species normalization, and hybrid retrieval."""

import argparse
import json
import sys
from pathlib import Path

import pymysql
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from llm_slots import SlotExtractor, Slots
from phase1_config import InventoryError, Settings


def load_catalog(settings: Settings) -> dict:
    try:
        connection = pymysql.connect(
            host=settings.db_host, port=settings.db_port, user=settings.db_user,
            password=settings.db_password, database=settings.db_name,
            charset="utf8mb4", connect_timeout=5, read_timeout=10,
        )
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT alias, canonical_species FROM species_aliases")
                aliases = dict(cursor.fetchall())
                cursor.execute("SELECT DISTINCT species FROM products ORDER BY species")
                species = [row[0] for row in cursor.fetchall()]
            return {"species": species, "aliases": aliases}
        finally:
            connection.close()
    except pymysql.MySQLError as exc:
        raise InventoryError("无法读取 MariaDB 商品目录；请启动容器并检查 DB_* 配置。") from exc


def build_filter(slots: Slots, *, include_generic_water=False) -> models.Filter:
    conditions = [models.FieldCondition(key="stock", range=models.Range(gt=0))]
    if slots.species is not None:
        conditions.append(models.FieldCondition(key="species", match=models.MatchValue(value=slots.species)))
    if slots.max_price is not None:
        conditions.append(models.FieldCondition(key="price", range=models.Range(lte=slots.max_price)))
    if slots.water_type is not None:
        water_match = (
            models.MatchAny(any=[slots.water_type, "通用"])
            if include_generic_water and slots.water_type != "通用"
            else models.MatchValue(value=slots.water_type)
        )
        conditions.append(models.FieldCondition(key="water_type", match=water_match))
    return models.Filter(must=conditions)


class InventoryTool:
    def __init__(self, settings=None, *, catalog=None, client=None, embedder=None, extractor=None):
        self.settings = settings or Settings.from_env()
        self.catalog = catalog if catalog is not None else load_catalog(self.settings)
        # The SDK's version probe ignores trust_env; use the pinned 1.15 client/server pair.
        self.client = client if client is not None else QdrantClient(
            url=self.settings.qdrant_url, trust_env=False, timeout=15, check_compatibility=False
        )
        self.embedder = embedder
        self.extractor = extractor or SlotExtractor(self.settings)

    def close(self):
        self.client.close()

    def query(self, query: str, limit: int = 10) -> dict:
        # Validate configuration before downloading/loading an embedding model.
        slots = self.extractor.extract(query, self.catalog)
        result = self.search(slots, limit=limit)
        return {"query": query, **result}

    def query_with_fallback(self, query: str, limit: int = 10, *, allow_budget_relaxation=True) -> dict:
        slots = self.extractor.extract(query, self.catalog)
        return {"query": query, **self.search_with_fallback(
            slots, limit, allow_budget_relaxation=allow_budget_relaxation
        )}

    def search_with_fallback(self, slots: Slots, limit: int = 10, *, allow_budget_relaxation=True) -> dict:
        from fallback_logic import search_with_fallback

        if not isinstance(allow_budget_relaxation, bool):
            raise InventoryError("allow_budget_relaxation 必须为布尔值。")
        canonical = self.catalog["aliases"].get(slots.species, slots.species)
        normalized = slots.model_copy(update={"species": canonical})
        return search_with_fallback(
            self.search, normalized, limit, allow_budget_relaxation=allow_budget_relaxation
        )

    def search(self, slots: Slots, limit: int = 10, *, include_generic_water=False) -> dict:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 50:
            raise InventoryError("返回数量 limit 必须为 1 至 50 的整数。")
        canonical = self.catalog["aliases"].get(slots.species, slots.species)
        normalized = slots.model_copy(update={"species": canonical})
        semantic_text = normalized.soft_query or "路亚装备"
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            try:
                self.embedder = SentenceTransformer(self.settings.embedding_model)
            except Exception as exc:
                raise InventoryError("无法加载 embedding 模型；请检查缓存、网络和 EMBEDDING_MODEL。") from exc
        try:
            vector = self.embedder.encode(semantic_text, normalize_embeddings=True).tolist()
            points = self.client.query_points(
                collection_name=self.settings.collection,
                query=vector, query_filter=build_filter(normalized, include_generic_water=include_generic_water),
                limit=limit, with_payload=True,
            ).points
        except Exception as exc:
            raise InventoryError("向量查询失败；请检查 Qdrant 服务、集合是否已同步及模型维度。") from exc
        return {
            "status": "ok" if points else "empty",
            "slots": normalized.model_dump(),
            "returned_count": len(points),
            "products": [
                {"id": point.id, "score": point.score, **(point.payload or {})}
                for point in points
            ],
        }


def print_result(result: dict, as_json: bool):
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return
    print("槽位：" + json.dumps(result["slots"], ensure_ascii=False))
    if "reply" in result:
        print(result["reply"])
        print(f"检索次数：{len(result['attempts'])}")
    if result["status"] == "empty":
        if "reply" not in result:
            print("当前条件下没有候选商品。")
        return
    for rank, product in enumerate(result["products"], start=1):
        print(f"{rank}. {product['name']} | {product['species']} | {product['water_type']} | "
              f"CNY {product['price']:.2f} | 库存 {product['stock']} | score={product['score']:.4f}")
        print("   " + product["description"])
        if product.get("deviations"):
            print("   注意：" + "；".join(product["deviations"]))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Phase 2 路亚库存查询工具（空结果降级）")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--query", help="自然语言选购需求，需要配置 LLM")
    modes.add_argument("--slots-file", type=Path, help="结构化 JSON 文件，不调用 LLM")
    modes.add_argument("--interactive", action="store_true", help="连续输入自然语言需求，输入 exit 退出")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="输出完整 JSON")
    parser.add_argument("--strict", action="store_true", help="使用 Phase 1 严格查询，不放宽任何条件")
    parser.add_argument("--no-budget-relaxation", action="store_true", help="不展示超预算替代品，仍允许通用水域")
    args = parser.parse_args()
    tool = None
    try:
        settings = Settings.from_env()
        if not args.slots_file:
            settings.require_llm()
        tool = InventoryTool(settings)
        def search_slots(slots):
            return (tool.search(slots, args.limit) if args.strict else tool.search_with_fallback(
                slots, args.limit, allow_budget_relaxation=not args.no_budget_relaxation
            ))

        def search_query(query):
            return (tool.query(query, args.limit) if args.strict else tool.query_with_fallback(
                query, args.limit, allow_budget_relaxation=not args.no_budget_relaxation
            ))

        if args.slots_file:
            slots = Slots.model_validate_json(args.slots_file.read_text(encoding="utf-8-sig"))
            print_result(search_slots(slots), args.json)
        elif args.query:
            print_result(search_query(args.query), args.json)
        else:
            print("请输入选购需求；每次输入独立查询。输入 exit 退出。")
            if not args.strict:
                print("空结果将尝试通用水域及预算替代方案；超预算商品会明确提示。")
            while True:
                query = input("需求> ").strip()
                if query.lower() in {"exit", "quit", "退出"}:
                    break
                if not query:
                    continue
                try:
                    print_result(search_query(query), args.json)
                except InventoryError as exc:
                    print(str(exc), file=sys.stderr)
        return 0
    except (InventoryError, ValidationError, OSError) as exc:
        if isinstance(exc, InventoryError):
            print(str(exc), file=sys.stderr)
        else:
            print("槽位文件无法读取或格式无效，请检查四个字段及其类型。", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, EOFError):
        return 0
    finally:
        if tool is not None:
            tool.close()


if __name__ == "__main__":
    raise SystemExit(main())
