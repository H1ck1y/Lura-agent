"""Phase 3 contracts: framework loop, read-only tools, grounding and real Qdrant."""

import contextlib
import copy
import io
import json
import unittest
from unittest.mock import Mock

import httpx
import numpy as np
from qdrant_client import QdrantClient, models

from agent_llm import AgentChatLLM
from knowledge_tool import KNOWLEDGE_COLLECTION, KnowledgeRAGTool, load_sources
from luya_agent import AuditedTool, LuyaAgent
from phase1_config import InventoryError, Settings
from rag_retrieval import RetrievalOptions


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def invoke(self, messages, **kwargs):
        self.requests.append(copy.deepcopy(messages))
        return next(self.responses)


def inventory_result(status="ok"):
    return {"status": status, "reply": "精确匹配" if status == "ok" else "预算替代，需要确认",
            "products": [{"id": 1, "name": "样例软饵", "price": 48, "stock": 2,
                          "deviations": ["超过原预算 8 元"] if status == "alternatives" else []}]}


class FakeRAG:
    name = "rag"
    description = "路亚知识检索"
    expandable = False

    def get_parameters(self):
        from hello_agents.tools.base import ToolParameter
        return [ToolParameter("query", "string", "知识问题", True)]

    def run(self, parameters):
        return json.dumps({"status": "ok", "reply": "片段", "snippets": [{
            "citation": "K1", "source_file": "02_jighead_snags.md", "title": "挂底风险",
            "url": "https://example.org/source", "content": "露出钩尖的铅头钩在障碍区有挂底风险。",
        }]}, ensure_ascii=False)

    def close(self):
        pass


class AgentTests(unittest.TestCase):
    def create(self, responses, *, result=None, rag=None, **kwargs):
        self.llm = ScriptedLLM(*responses)
        self.inventory = Mock()
        self.inventory.query_with_fallback.return_value = result or inventory_result()
        self.inventory.query.return_value = inventory_result()
        self.app = LuyaAgent(Settings(), llm=self.llm, inventory=self.inventory,
                             rag=rag or FakeRAG(), **kwargs)
        self.addCleanup(self.app.close)
        return self.app

    def test_inventory_registered_and_invoked_by_framework_loop(self):
        app = self.create(['[TOOL_CALL:inventory:{"query":"钓鲈鱼，预算100"}]', '样例软饵可供参考。'])
        result = app.ask("钓鲈鱼，预算100")
        self.assertEqual(set(app.registry.list_tools()), {"inventory", "rag"})
        self.assertEqual([e["tool"] for e in result["tool_calls"]], ["inventory"])
        self.assertIn("48.00 元，库存 2", result["answer"])
        self.assertIn("工具执行结果", self.llm.requests[1][-1]["content"])

    def test_knowledge_uses_rag_without_inventory(self):
        app = self.create(['[TOOL_CALL:rag:{"query":"铅头钩为什么挂底"}]', '露出钩尖有风险。[K1]'])
        result = app.ask("铅头钩为什么挂底")
        self.inventory.query_with_fallback.assert_not_called()
        self.assertEqual([e["tool"] for e in result["tool_calls"]], ["rag"])
        self.assertEqual(result["sources"][0]["citation"], "K1")
        self.assertIn("知识来源", result["answer"])

    def test_mixed_question_can_use_both_tools(self):
        app = self.create(['[TOOL_CALL:inventory:{"query":"鲈鱼100"}]\n[TOOL_CALL:rag:{"query":"匀收技巧"}]', '先试简单匀收。'])
        result = app.ask("推荐预算100的鲈鱼饵，并说说匀收技巧")
        self.assertEqual({e["tool"] for e in result["tool_calls"]}, {"inventory", "rag"})

    def test_original_constraints_not_lost_by_router(self):
        app = self.create(['[TOOL_CALL:inventory:{"query":"钓鲈鱼"}]', '候选见下。'])
        question = "钓鲈鱼，浑水，预算40元，远投"
        app.ask(question)
        self.inventory.query_with_fallback.assert_called_once_with(question, 5, allow_budget_relaxation=True)

    def test_budget_option_and_strict_inventory(self):
        app = self.create(['[TOOL_CALL:inventory:query=鲈鱼]','候选。'], allow_budget_relaxation=False)
        app.ask("鲈鱼预算100")
        self.assertFalse(self.inventory.query_with_fallback.call_args.kwargs["allow_budget_relaxation"])
        app = self.create(['[TOOL_CALL:inventory:query=鲈鱼]','候选。'], strict_inventory=True)
        app.ask("鲈鱼预算100")
        self.inventory.query.assert_called_once_with("鲈鱼预算100", 5)
        self.inventory.query_with_fallback.assert_not_called()

    def test_relaxation_notice_survives_generator_omission(self):
        app = self.create(['[TOOL_CALL:inventory:query=黑鱼]','这款可参考。'], result=inventory_result("alternatives"))
        result = app.ask("黑鱼预算40")
        self.assertIn("超过原预算 8 元", result["answer"])
        self.assertIn("需要确认", result["answer"])

    def test_no_evidence_cannot_be_replaced_by_invented_answer(self):
        rag = Mock(wraps=FakeRAG())
        rag.name = "rag"
        rag.description = "知识"
        rag.run.return_value = json.dumps({"status": "empty", "snippets": [], "reply": "资料不足，不能回答。"})
        app = self.create(['[TOOL_CALL:rag:{"query":"未知钓法"}]','虚构事实。'], rag=rag)
        result = app.ask("未知钓法怎么操作")
        self.assertEqual(result["answer"], "资料不足，不能回答。")
        self.assertEqual(result["status"], "no_evidence")

    def test_service_error_not_reported_as_empty_stock(self):
        app = self.create(['[TOOL_CALL:inventory:query=鲈鱼]','假装成功。'])
        self.inventory.query_with_fallback.side_effect = InventoryError("库存服务失败")
        result = app.ask("鲈鱼100")
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["answer"], "库存服务失败")

    def test_third_party_error_does_not_leak_secret(self):
        app = self.create(['[TOOL_CALL:inventory:query=鲈鱼]','假装成功。'])
        self.inventory.query_with_fallback.side_effect = RuntimeError("secret-test-key")
        result = app.ask("鲈鱼100")
        self.assertNotIn("secret-test-key", json.dumps(result))

    def test_greeting_has_no_tool_calls_and_history_is_cleared(self):
        app = self.create(['你好，我提供路亚咨询。', '每次问题独立。'])
        app.ask("你好，第一轮独有信息")
        result = app.ask("第二轮")
        self.assertEqual(result["tool_calls"], [])
        self.assertNotIn("第一轮独有信息", json.dumps(self.llm.requests[-1], ensure_ascii=False))
        self.assertEqual(app.agent._history, [])

    def test_parser_preserves_json_brackets_commas_and_quotes(self):
        query = '怎么理解 [停顿], query=技巧，"米诺"'
        marker = '[TOOL_CALL:rag:' + json.dumps({"query": query}, ensure_ascii=False) + ']'
        app = self.create([marker, '见资料。'])
        calls = app.agent._parse_tool_calls(marker)
        self.assertEqual(app.agent._parse_tool_parameters("rag", calls[0]["parameters"]), {"query": query})
        self.assertEqual(calls[0]["original"], marker)
        self.assertEqual(app.agent._parse_tool_parameters("rag", "query=a,b=c"), {"query": "a,b=c"})

    def test_unknown_tools_and_malformed_markers_rejected(self):
        for marker in ['[TOOL_CALL:clear:{"query":"删除"}]', '[TOOL_CALL:rag:{bad}]', '[TOOL_CALL:rag:query=x']:
            app = self.create([marker])
            with self.subTest(marker=marker), self.assertRaises(InventoryError):
                app.ask("测试")
            self.assertEqual(app.agent._history, [])

    def test_no_more_than_one_execution_per_tool(self):
        app = self.create(['[TOOL_CALL:inventory:query=鲈鱼]' * 2, '已查询。'])
        result = app.ask("鲈鱼")
        self.assertEqual(len(result["tool_calls"]), 1)
        self.inventory.query_with_fallback.assert_called_once()

    def test_iteration_limit_does_not_return_unexecuted_marker(self):
        marker = '[TOOL_CALL:inventory:query=鲈鱼]'
        app = self.create([marker] * 4)
        with self.assertRaisesRegex(InventoryError, "有限步骤"):
            app.ask("鲈鱼")
        self.inventory.query_with_fallback.assert_called_once()

    def test_invalid_input_rejected_without_model_call(self):
        app = self.create([])
        for question in ["", " ", "x" * 4001, None]:
            with self.subTest(question=str(question)[:10]), self.assertRaises(InventoryError):
                app.ask(question)
        self.assertEqual(self.llm.requests, [])


class TinyEmbedder:
    def encode(self, texts, **kwargs):
        def one(text):
            if "量子" in text:
                return [0.0, 1.0]
            return [-1.0, 0.0] if "颜色" in text else [1.0, 0.0]
        return np.array(one(texts) if isinstance(texts, str) else [one(t) for t in texts])


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.tool = KnowledgeRAGTool(Settings(), client=self.client, embedder=TinyEmbedder(),
                                     options=RetrievalOptions(mode="basic"))
        self.addCleanup(self.tool.close)

    def index(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.tool.index_documents()

    def test_real_qdrant_index_retrieve_citations(self):
        info = self.index()
        self.assertEqual(info["documents"], 5)
        result = json.loads(self.tool.run({"query": "铅头钩挂底"}))
        self.assertEqual(result["status"], "ok")
        self.assertTrue(all(s["source_file"] in self.tool.sources for s in result["snippets"]))
        self.assertTrue(all(s["url"].startswith("https://") for s in result["snippets"]))

    def test_idempotent_upsert_and_inventory_collection_isolated(self):
        self.client.create_collection("luya_products", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
        first = self.index()
        before = self.client.count(KNOWLEDGE_COLLECTION).count
        second = self.index()
        self.assertEqual(first["corpus_revision"], second["corpus_revision"])
        self.assertEqual(self.client.count(KNOWLEDGE_COLLECTION).count, before)
        self.assertEqual(self.client.count("luya_products").count, 0)

    def test_missing_index_is_service_error(self):
        with self.assertRaisesRegex(InventoryError, "尚未索引"):
            self.tool.search_result("铅头钩")

    def test_unrelated_query_is_empty_not_fabricated(self):
        self.index()
        result = self.tool.search_result("量子计算机")
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["snippets"], [])
        self.assertIn("没有足够", result["reply"])

    def test_write_clear_and_arbitrary_file_operations_not_exposed(self):
        for parameters in [{"action": "clear", "confirm": True}, {"query": "x", "file_path": ".env"},
                           {"action": "add_document", "file_path": ".env"}]:
            with self.subTest(parameters=parameters), self.assertRaises(InventoryError):
                self.tool.run(parameters)

    def test_foreign_namespace_and_old_revision_not_retrieved(self):
        self.index()
        self.client.upsert(KNOWLEDGE_COLLECTION, [models.PointStruct(
            id=100, vector=[1.0, 0.0], payload={
                "rag_namespace": "foreign", "corpus_revision": self.tool.revision,
                "embedding_model": self.tool.settings.embedding_model,
                "source_file": "02_jighead_snags.md", "content": "伪造资料",
            }
        ), models.PointStruct(id=101, vector=[1.0, 0.0], payload={
            "rag_namespace": "luya_phase3", "corpus_revision": "old",
            "embedding_model": self.tool.settings.embedding_model,
            "source_file": "02_jighead_snags.md", "content": "过期资料",
        })])
        result = self.tool.search_result("挂底")
        self.assertFalse(any(s["content"] in {"伪造资料", "过期资料"} for s in result["snippets"]))

    def test_database_failure_not_empty_result(self):
        self.index()
        self.client.query_points = Mock(side_effect=RuntimeError("offline"))
        with self.assertRaisesRegex(InventoryError, "查询失败"):
            self.tool.search_result("挂底")

    def test_changed_corpus_requires_reindex_not_empty_answer(self):
        self.index()
        self.tool.revision = "changed-revision"
        with self.assertRaisesRegex(InventoryError, "已变更"):
            self.tool.search_result("挂底")

    def test_invalid_vectors_do_not_get_indexed(self):
        self.tool.embedder.encode = Mock(return_value=np.full((5, 2), float("nan")))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(InventoryError, "无效向量"):
            self.tool.index_documents()
        self.assertFalse(self.client.collection_exists(KNOWLEDGE_COLLECTION))

    def test_knowledge_collection_cannot_equal_inventory_collection(self):
        with self.assertRaisesRegex(InventoryError, "必须与商品集合不同"):
            KnowledgeRAGTool(Settings(collection=KNOWLEDGE_COLLECTION))

    def test_manifest_does_not_allow_path_traversal_or_secret_files(self):
        from unittest.mock import patch
        for name in ["../README.md", ".env", "../knowledge_base/02_jighead_snags.md"]:
            with self.subTest(name=name), patch("pathlib.Path.read_text", return_value=json.dumps({
                name: {"title": "unsafe", "url": None}
            })), self.assertRaises(InventoryError):
                load_sources()


class AgentLLMTests(unittest.TestCase):
    def settings(self):
        return Settings(llm_base_url="https://unit-test.invalid/v1", llm_model="configured-model",
                        llm_api_key="unit-secret", llm_extra_body={"reasoning_effort": "low"})

    def test_plain_text_request_preserves_model_reasoning_without_native_tools(self):
        def respond(request):
            self.request = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": "你好"}, "finish_reason": "stop"}]})
        client = AgentChatLLM(self.settings(), httpx.MockTransport(respond))
        self.assertEqual(client.invoke([{"role": "user", "content": "你好"}]), "你好")
        self.assertEqual(self.request["model"], "configured-model")
        self.assertEqual(self.request["reasoning_effort"], "low")
        for name in ["temperature", "max_tokens", "response_format", "tools"]:
            self.assertNotIn(name, self.request)

    def test_http_errors_and_malformed_response_safe(self):
        for status, response in [(401, {"error": "unit-secret"}), (200, {"choices": [None]}),
                                 (200, {"choices": [{"message": {"content": None}}]})]:
            llm = AgentChatLLM(self.settings(), httpx.MockTransport(lambda request: httpx.Response(status, json=response)))
            with self.subTest(status=status), self.assertRaises(InventoryError) as caught:
                llm.invoke([])
            self.assertNotIn("unit-secret", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
