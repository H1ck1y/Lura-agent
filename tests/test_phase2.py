"""Phase 2 uses real in-memory Qdrant, not mocked retrieval results."""

import unittest
from unittest.mock import Mock

from qdrant_client import QdrantClient, models

from fallback_logic import expanded_budget
from inventory_tool import InventoryTool
from llm_slots import Slots
from phase1_config import InventoryError, Settings
from test_phase1 import FixedEmbedder, slots


class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            "luya_products", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
        )
        records = [
            (1, "鲈鱼", "清水", 100, 5, [1.0, 0.0]),
            (2, "鲈鱼", "通用", 90, 5, [0.8, 0.6]),
            (3, "鲈鱼", "清水", 120, 5, [0.6, 0.8]),
            (4, "鲈鱼", "咸水", 10, 5, [1.0, 0.0]),
            (5, "鲈鱼", "通用", 10, 0, [1.0, 0.0]),
            (6, "黑鱼", "浑水", 10, 5, [1.0, 0.0]),
            (7, "鳡鱼", "通用", 120, 5, [1.0, 0.0]),
            (8, "鳡鱼", "清水", 120.01, 5, [1.0, 0.0]),
            (9, "青甘", "咸水", 120, 5, [1.0, 0.0]),
            (10, "青甘", "清水", 200, 5, [1.0, 0.0]),
        ]
        self.client.upsert("luya_products", [
            models.PointStruct(id=id_, vector=vector, payload={
                "species": species, "water_type": water, "price": price, "stock": stock,
                "name": f"fixture-{id_}", "description": "description",
            }) for id_, species, water, price, stock, vector in records
        ])
        self.tool = InventoryTool(
            Settings(), client=self.client, embedder=FixedEmbedder(),
            catalog={"species": ["鲈鱼", "黑鱼", "鳡鱼", "青甘"], "aliases": {"水老虎": "鳡鱼"}},
        )

    def tearDown(self):
        self.tool.close()

    def test_exact_hit_stops_without_relaxation(self):
        result = self.tool.search_with_fallback(slots(water_type="清水"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["attempts"]), 1)
        self.assertEqual(result["relaxations"], [])
        self.assertEqual([p["id"] for p in result["products"]], [1])
        self.assertTrue(result["products"][0]["matches_original_filters"])

    def test_generic_water_before_budget(self):
        result = self.tool.search_with_fallback(slots(water_type="浑水"))
        self.assertEqual(result["status"], "alternatives")
        self.assertEqual([a["stage"] for a in result["attempts"]], ["strict", "generic_water"])
        self.assertEqual([p["id"] for p in result["products"]], [2])
        self.assertEqual(result["attempts"][-1]["water_types"], ["浑水", "通用"])
        self.assertEqual(result["products"][0]["relaxed_fields"], ["water_type"])
        self.assertIn("需确认实际适用性", result["reply"])

    def test_budget_only_without_unspecified_water_filter(self):
        result = self.tool.search_with_fallback(slots(species="青甘", max_price=100))
        self.assertEqual([a["stage"] for a in result["attempts"]], ["strict", "budget"])
        self.assertIsNone(result["attempts"][-1]["water_types"])
        self.assertEqual(result["slots"]["max_price"], 100)
        self.assertEqual(result["effective_slots"]["max_price"], 120)
        self.assertEqual([p["id"] for p in result["products"]], [9])
        self.assertEqual(result["products"][0]["relaxed_fields"], ["max_price"])
        self.assertIn("超过原预算", result["reply"])
        self.assertIn("20 元", result["products"][0]["deviations"][0])

    def test_combined_relaxation_and_twenty_percent_boundary(self):
        original = slots(species="水老虎", water_type="浑水")
        snapshot = original.model_dump()
        result = self.tool.search_with_fallback(original)
        self.assertEqual([a["stage"] for a in result["attempts"]], ["strict", "generic_water", "budget"])
        self.assertEqual([p["id"] for p in result["products"]], [7])
        self.assertEqual(result["slots"]["species"], "鳡鱼")
        self.assertEqual(result["products"][0]["relaxed_fields"], ["water_type", "max_price"])
        self.assertEqual(original.model_dump(), snapshot)

    def test_original_water_still_available_on_budget_retry(self):
        result = self.tool.search_with_fallback(slots(water_type="清水", max_price=75))
        self.assertEqual([p["id"] for p in result["products"]], [2])
        self.assertEqual(result["attempts"][-1]["water_types"], ["清水", "通用"])

    def test_unknown_species_is_not_replaced(self):
        result = self.tool.search_with_fallback(slots(species="不存在鱼种", water_type="清水"))
        self.assertEqual(result["status"], "empty")
        self.assertEqual(len(result["attempts"]), 3)
        self.assertTrue(all(a["slots"]["species"] == "不存在鱼种" for a in result["attempts"]))
        self.assertIn("确认鱼种名称", result["reply"])
        self.assertEqual(result["products"], [])

    def test_stock_and_species_never_dropped(self):
        result = self.tool.search_with_fallback(slots(water_type="浑水", max_price=10))
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["returned_count"], 0)

    def test_salt_water_not_substituted_for_fresh_water(self):
        result = self.tool.search_with_fallback(slots(species="青甘", water_type="清水"))
        self.assertEqual(result["status"], "empty")
        self.assertEqual(len(result["attempts"]), 3)
        self.assertNotIn(9, [p["id"] for p in result["products"]])

    def test_zero_budget_not_expanded(self):
        result = self.tool.search_with_fallback(slots(max_price=0))
        self.assertEqual(result["status"], "empty")
        self.assertEqual(len(result["attempts"]), 1)
        self.assertNotIn("有限放宽", result["reply"])

    def test_missing_budget_not_invented(self):
        result = self.tool.search_with_fallback(slots(species="unknown", max_price=None, water_type="清水"))
        self.assertEqual(len(result["attempts"]), 2)
        self.assertTrue(all(a["slots"]["max_price"] is None for a in result["attempts"]))

    def test_generic_water_does_not_create_duplicate_retry(self):
        result = self.tool.search_with_fallback(slots(water_type="通用", max_price=80))
        self.assertEqual([a["stage"] for a in result["attempts"]], ["strict", "budget"])
        self.assertEqual(result["attempts"][-1]["water_types"], ["通用"])

    def test_budget_relaxation_can_be_disabled(self):
        result = self.tool.search_with_fallback(slots(species="青甘"), allow_budget_relaxation=False)
        self.assertEqual(result["status"], "empty")
        self.assertEqual(len(result["attempts"]), 1)
        with self.assertRaises(InventoryError):
            self.tool.search_with_fallback(slots(), allow_budget_relaxation="false")

    def test_soft_query_preserved_on_every_attempt(self):
        original = slots(species="unknown", water_type="清水", soft_query="新手，不易挂底，不要沉重")
        result = self.tool.search_with_fallback(original)
        self.assertTrue(all(a["slots"]["soft_query"] == original.soft_query for a in result["attempts"]))

    def test_query_extracts_once_only(self):
        extracted = slots(species="水老虎", water_type="浑水")
        self.tool.extractor = Mock()
        self.tool.extractor.extract.return_value = extracted
        result = self.tool.query_with_fallback("钓水老虎，浑水，预算100")
        self.tool.extractor.extract.assert_called_once()
        self.assertEqual(result["query"], "钓水老虎，浑水，预算100")
        self.assertEqual(len(result["attempts"]), 3)

    def test_query_extraction_error_is_not_empty_result(self):
        self.tool.extractor = Mock()
        self.tool.extractor.extract.side_effect = InventoryError("LLM 失败")
        with self.assertRaisesRegex(InventoryError, "LLM 失败"):
            self.tool.query_with_fallback("test")

    def test_database_error_not_retried_as_empty(self):
        self.client.query_points = Mock(side_effect=RuntimeError("service offline"))
        with self.assertRaisesRegex(InventoryError, "向量查询失败"):
            self.tool.search_with_fallback(slots())
        self.client.query_points.assert_called_once()

    def test_limit_and_ranking_preserved(self):
        result = self.tool.search_with_fallback(slots(species=None, max_price=None), limit=2)
        self.assertEqual(result["returned_count"], 2)
        scores = [p["score"] for p in result["products"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for value in [0, 51, True]:
            with self.subTest(value=value), self.assertRaises(InventoryError):
                self.tool.search_with_fallback(slots(), limit=value)

    def test_phase1_api_remains_strict(self):
        result = self.tool.search(slots(species="青甘"))
        self.assertEqual(result["status"], "empty")
        self.assertNotIn("attempts", result)

    def test_budget_rounds_down_not_beyond_twenty_percent(self):
        self.assertEqual(expanded_budget(40), 48)
        self.assertEqual(expanded_budget(33.33), 39.99)
        for value in [None, 0, 0.001, 1.7e308]:
            with self.subTest(value=value):
                self.assertIsNone(expanded_budget(value))


if __name__ == "__main__":
    unittest.main()
