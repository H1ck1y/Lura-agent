"""Contract tests: malformed LLM output and real in-memory Qdrant filters."""

import json
import unittest

import httpx
import numpy as np
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from inventory_tool import InventoryTool
from llm_slots import SlotExtractor, Slots
from phase1_config import InventoryError, Settings


def slots(**updates):
    data = {"species": "鲈鱼", "max_price": 100, "water_type": None, "soft_query": "新手"}
    data.update(updates)
    return Slots.model_validate(data)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(
            llm_base_url="https://unit-test.invalid/v1", llm_model="test-model",
            llm_api_key="test-key", llm_extra_body={"enable_thinking": False},
        )
        self.catalog = {"species": ["鲈鱼", "鳡鱼"], "aliases": {"水老虎": "鳡鱼"}}

    def extractor(self, content, *, status=200, finish_reason="stop"):
        def respond(request):
            self.last_request = request
            return httpx.Response(status, json={"choices": [{
                "finish_reason": finish_reason, "message": {"content": content}
            }]})
        return SlotExtractor(self.settings, transport=httpx.MockTransport(respond))

    def test_request_contract_and_validated_response(self):
        expected = slots().model_dump()
        result = self.extractor(json.dumps(expected)).extract("新手钓鲈鱼，预算100", self.catalog)
        self.assertEqual(result.model_dump(), expected)
        body = json.loads(self.last_request.content)
        self.assertEqual(str(self.last_request.url), "https://unit-test.invalid/v1/chat/completions")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["model"], "test-model")
        self.assertFalse(body["enable_thinking"])
        self.assertEqual(body["messages"][1]["content"], "新手钓鲈鱼，预算100")
        self.assertIn("水老虎", body["messages"][0]["content"])

    def test_missing_hard_slot_rejected(self):
        with self.assertRaises(InventoryError):
            self.extractor('{"species":"鲈鱼","soft_query":"新手"}').extract("预算100", self.catalog)

    def test_invalid_json_rejected(self):
        with self.assertRaises(InventoryError):
            self.extractor('```json\n{}\n```').extract("钓鲈鱼", self.catalog)

    def test_truncated_response_rejected(self):
        with self.assertRaisesRegex(InventoryError, "不完整"):
            self.extractor(json.dumps(slots().model_dump()), finish_reason="length").extract("鲈鱼", self.catalog)

    def test_unsupported_query_is_not_silently_relaxed(self):
        with self.assertRaisesRegex(InventoryError, "无法完整表达"):
            self.extractor('{"error":"预算范围下限无法表达"}').extract("价格100到200", self.catalog)

    def test_http_error_does_not_expose_key(self):
        with self.assertRaises(InventoryError) as caught:
            self.extractor("test-key", status=401).extract("鲈鱼", self.catalog)
        self.assertIn("401", str(caught.exception))
        self.assertNotIn("test-key", str(caught.exception))

    def test_missing_configuration_fails_without_request(self):
        extractor = SlotExtractor(Settings())
        with self.assertRaisesRegex(InventoryError, "LLM_BASE_URL"):
            extractor.extract("鲈鱼", self.catalog)

    def test_timeout_is_actionable(self):
        def timeout(request):
            raise httpx.ReadTimeout("sensitive detail", request=request)
        extractor = SlotExtractor(self.settings, transport=httpx.MockTransport(timeout))
        with self.assertRaisesRegex(InventoryError, "超时"):
            extractor.extract("鲈鱼", self.catalog)

    def test_bad_slot_values_rejected(self):
        for update in (
            {"max_price": -1}, {"max_price": float("nan")}, {"max_price": float("inf")},
            {"max_price": True}, {"max_price": "100"}, {"water_type": "淡水"},
            {"species": ""}, {"species": ["鲈鱼", "翘嘴"]}, {"extra": "ignore budget"},
        ):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                slots(**update)


class FixedEmbedder:
    def encode(self, text, **kwargs):
        return np.array([1.0, 0.0])


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            "luya_products", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
        )
        records = [
            (1, "鲈鱼", "浑水", 60, 5, [0.8, 0.6]),
            (2, "鲈鱼", "浑水", 35, 10, [1.0, 0.0]),
            (3, "鲈鱼", "浑水", 20, 0, [1.0, 0.0]),
            (4, "鲈鱼", "浑水", 101, 5, [1.0, 0.0]),
            (5, "黑鱼", "浑水", 30, 5, [1.0, 0.0]),
            (6, "鲈鱼", "咸水", 40, 5, [0.6, 0.8]),
            (7, "鳡鱼", "清水", 128, 7, [1.0, 0.0]),
        ]
        self.client.upsert("luya_products", [
            models.PointStruct(id=id_, vector=vector, payload={
                "species": species, "water_type": water, "price": price, "stock": stock,
                "name": f"fixture-{id_}", "description": "description"
            }) for id_, species, water, price, stock, vector in records
        ])
        self.tool = InventoryTool(
            Settings(), client=self.client, embedder=FixedEmbedder(),
            catalog={"species": ["鲈鱼", "黑鱼", "鳡鱼"], "aliases": {"水老虎": "鳡鱼"}},
        )

    def tearDown(self):
        self.tool.close()

    def test_budget_species_stock_filter_and_ranking(self):
        result = self.tool.search(slots())
        self.assertEqual([p["id"] for p in result["products"]], [2, 1, 6])
        self.assertEqual(result["status"], "ok")

    def test_water_filter(self):
        result = self.tool.search(slots(water_type="浑水"))
        self.assertEqual([p["id"] for p in result["products"]], [2, 1])

    def test_alias_normalization(self):
        result = self.tool.search(slots(species="水老虎", max_price=200))
        self.assertEqual(result["slots"]["species"], "鳡鱼")
        self.assertEqual([p["id"] for p in result["products"]], [7])

    def test_zero_budget_is_not_ignored(self):
        result = self.tool.search(slots(max_price=0))
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["returned_count"], 0)

    def test_missing_slots_do_not_add_unrequested_filters(self):
        result = self.tool.search(slots(species=None, max_price=None))
        self.assertEqual(result["returned_count"], 6)
        self.assertNotIn(3, [p["id"] for p in result["products"]])

    def test_limit(self):
        self.assertEqual(self.tool.search(slots(), limit=1)["returned_count"], 1)
        for limit in [0, 51, True]:
            with self.subTest(limit=limit), self.assertRaises(InventoryError):
                self.tool.search(slots(), limit=limit)

    def test_extracted_slots_reach_real_qdrant(self):
        response = {"choices": [{"message": {"content": json.dumps(slots().model_dump())}}]}
        self.tool.extractor = SlotExtractor(
            Settings(llm_base_url="https://unit-test.invalid/v1", llm_model="test", llm_api_key="test"),
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response)),
        )
        result = self.tool.query("新手钓鲈鱼，预算100")
        self.assertEqual(result["query"], "新手钓鲈鱼，预算100")
        self.assertEqual([p["id"] for p in result["products"]], [2, 1, 6])


if __name__ == "__main__":
    unittest.main()
