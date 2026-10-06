"""Deterministic hybrid retrieval contracts; fake networks/models, real Qdrant."""

import argparse
import contextlib
import io
import json
import os
import unittest
from unittest.mock import Mock, patch

import httpx
from qdrant_client import QdrantClient, models

from agent_llm import AgentChatLLM
from evaluate_rag import metrics
from knowledge_tool import KnowledgeRAGTool, KNOWLEDGE_COLLECTION
from phase1_config import InventoryError, Settings
from rag_retrieval import (ChineseBM25, QueryExpansion, RetrievalOptions,
                           add_retrieval_arguments, options_from_arguments, reciprocal_rank_fusion)
from test_phase3 import TinyEmbedder, ScriptedLLM


class ComponentTests(unittest.TestCase):
    def test_evaluation_metrics_with_positive_and_negative_cases(self):
        detail = {"elapsed_ms": 10, "extra_llm_calls": 0, "extra_prompt_tokens": 0,
                  "extra_completion_tokens": 0, "usage_complete": True, "warnings": []}
        result = metrics([
            {"relevant": ["a"], "retrieved": ["b", "a"], "status": "ok", "retrieval": detail},
            {"relevant": [], "retrieved": [], "status": "empty", "retrieval": detail},
        ])
        self.assertEqual(result["hit_at_1"], 0)
        self.assertEqual(result["recall_at_3"], 1)
        self.assertEqual(result["mrr_at_3"], 0.5)
        self.assertEqual(result["negative_abstention_rate"], 1)
        self.assertEqual(result["median_elapsed_ms"], 10)

    def test_evaluation_empty_groups_not_reported_as_perfect(self):
        result = metrics([])
        self.assertIsNone(result["recall_at_3"])
        self.assertIsNone(result["negative_abstention_rate"])

    def test_rrf_fuses_ranks_not_raw_scores_and_deduplicates(self):
        a = {"id": "a", "score": 999, "metadata": {}}
        b = {"id": "b", "score": 0.1, "metadata": {}}
        hits = reciprocal_rank_fusion([("dense", [b, a]), ("bm25", [a, a])])
        self.assertEqual([h["id"] for h in hits], ["a", "b"])
        self.assertAlmostEqual(hits[0]["score"], 1 / 62 + 1 / 61)
        self.assertEqual(hits[0]["scores"]["bm25"], 999)

    def test_chinese_bm25_has_no_score_for_nonmatches(self):
        index = ChineseBM25([
            {"id": "1", "metadata": {"content": "铅头钩露钩尖，障碍区挂底。"}},
            {"id": "2", "metadata": {"content": "清水选自然色，浑水试荧光色。"}},
        ])
        self.assertEqual(index.search("铅头钩为什么挂底", 3)[0]["id"], "1")
        self.assertEqual(index.search("量子计算机纠错", 3), [])

    def test_invalid_options(self):
        for kwargs in [{"mode": "x"}, {"candidate_k": 2}, {"candidate_k": True},
                       {"min_rerank_score": float("nan")}, {"min_rerank_score": 2},
                       {"mode": "basic", "hyde": True}, {"reranker_model": " "}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(InventoryError):
                RetrievalOptions(**kwargs)

    def test_env_defaults_optional_features_off(self):
        with patch.dict(os.environ, {}, clear=True):
            options = RetrievalOptions.from_env()
        self.assertEqual(options.mode, "hybrid")
        self.assertTrue(options.rerank)
        self.assertFalse(options.mqe or options.hyde)

    def test_invalid_env_configuration(self):
        for key, value in [("RAG_MQE", "maybe"), ("RAG_CANDIDATE_K", "abc"),
                           ("RAG_MIN_RERANK_SCORE", "nan")]:
            with patch.dict(os.environ, {key: value}, clear=True), self.assertRaises(InventoryError):
                RetrievalOptions.from_env()

    def test_cli_basic_disables_env_expansions_and_cli_off_overrides_env(self):
        parser = argparse.ArgumentParser()
        add_retrieval_arguments(parser)
        with patch.dict(os.environ, {"RAG_MQE": "true", "RAG_HYDE": "true"}, clear=True):
            options = options_from_arguments(parser.parse_args(["--rag-mode", "basic"]))
            self.assertFalse(options.mqe or options.hyde)
            options = options_from_arguments(parser.parse_args(["--no-mqe", "--no-hyde", "--no-rerank"]))
            self.assertFalse(options.mqe or options.hyde or options.rerank)

    def test_expansion_deduplicates_and_preserves_original_query(self):
        llm = ScriptedLLM('{"queries":["原问题","新措辞"]}')
        self.assertEqual(QueryExpansion(llm).generate("原问题", "mqe"), ["新措辞"])
        self.assertEqual(llm.requests[0][1]["content"], "原问题")

    def test_bad_expansion_json_rejected(self):
        for raw, kind in [('[]', 'mqe'), ('{"queries":"x"}', 'mqe'),
                          ('{"queries":["a","b","c"]}', 'mqe'),
                          ('{"queries":[1]}', 'mqe'), ('{"queries":["x"],"url":"bad"}', 'mqe'),
                          ('{"hypothesis":""}', 'hyde'), ('{"hypothesis":null}', 'hyde'),
                          (json.dumps({"hypothesis": "x" * 601}), 'hyde')]:
            with self.subTest(raw=raw), self.assertRaises((ValueError, TypeError)):
                QueryExpansion(ScriptedLLM(raw)).generate("问题", kind)


class HybridKnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.ranker = Mock()
        self.ranker.score.side_effect = lambda q, hits: [0.9 if h["metadata"]["source_file"] == "02_jighead_snags.md" else 0.2 for h in hits]
        self.tool = KnowledgeRAGTool(Settings(), client=self.client, embedder=TinyEmbedder(),
                                     options=RetrievalOptions(), reranker=self.ranker)
        self.addCleanup(self.tool.close)
        with contextlib.redirect_stdout(io.StringIO()):
            self.tool.index_documents()

    def test_default_pipeline_retrieves_and_reranks_real_chunks(self):
        result = self.tool.search_result("铅头钩为什么容易挂底")
        self.assertEqual(result["snippets"][0]["source_file"], "02_jighead_snags.md")
        snippet = result["snippets"][0]
        self.assertEqual(snippet["score_type"], "reranker_sigmoid")
        self.assertIn("bm25", snippet["scores"])
        self.assertIn("rrf", snippet["scores"])
        self.assertTrue(result["retrieval"]["rerank_applied"])
        self.assertEqual(result["retrieval"]["extra_llm_calls"], 0)
        self.ranker.score.assert_called_once()

    def test_hybrid_without_rerank_is_explicit_not_fake_rerank(self):
        self.tool.options = RetrievalOptions(rerank=False)
        result = self.tool.search_result("铅头钩挂底")
        self.assertEqual(result["snippets"][0]["score_type"], "rrf")
        self.assertFalse(result["retrieval"]["rerank_applied"])
        self.ranker.score.assert_not_called()

    def test_bm25_rescues_lexical_match_below_dense_threshold(self):
        self.tool.embedder.encode = Mock(return_value=__import__("numpy").array([0.0, 1.0]))
        result = self.tool.search_result("铅头钩挂底")
        self.assertEqual(result["snippets"][0]["source_file"], "02_jighead_snags.md")
        self.assertNotIn("dense", result["snippets"][0]["scores"])

    def test_no_evidence_skips_optional_calls_and_reranker(self):
        self.tool.options = RetrievalOptions(mqe=True, hyde=True)
        self.tool.expander = Mock()
        result = self.tool.search_result("量子计算机纠错")
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["retrieval"]["extra_llm_calls"], 0)
        self.tool.expander.generate.assert_not_called()
        self.ranker.score.assert_not_called()

    def test_two_features_max_two_calls_hypothesis_never_becomes_evidence(self):
        llm = ScriptedLLM('{"queries":["钩尖暴露风险","铅头钩配软饵"]}',
                          '{"hypothesis":"假设文本标记XYZ：铅头钩易挂底。"}')
        self.tool.options = RetrievalOptions(mqe=True, hyde=True)
        self.tool.expansion_llm = llm
        self.tool.expander = QueryExpansion(llm)
        result = self.tool.search_result("铅头钩挂底")
        self.assertEqual(result["retrieval"]["extra_llm_calls"], 2)
        self.assertEqual(result["retrieval"]["mqe_generated_count"], 2)
        self.assertEqual(len(llm.requests), 2)
        self.assertNotIn("假设文本标记XYZ", json.dumps(result, ensure_ascii=False))
        self.assertFalse(result["retrieval"]["usage_complete"])

    def test_expansion_can_recall_new_real_chunk_reranked_against_original_query(self):
        llm = ScriptedLLM('{"queries":["颜色"]}')
        self.tool.options = RetrievalOptions(mqe=True)
        self.tool.expansion_llm = llm
        self.tool.expander = QueryExpansion(llm)
        query = "铅头钩挂底"
        self.tool.search_result(query)
        called_query, candidates = self.ranker.score.call_args.args
        self.assertEqual(called_query, query)
        expanded = [h for h in candidates if h["metadata"]["source_file"] == "04_water_color.md"]
        self.assertTrue(expanded)
        self.assertIn("mqe_dense_1", expanded[0]["channels"])
        self.assertNotIn("dense", expanded[0]["channels"])

    def test_agent_mixed_question_preserves_inventory_and_isolates_knowledge_query(self):
        from luya_agent import LuyaAgent
        from test_phase3 import inventory_result, FakeRAG
        rag = Mock(wraps=FakeRAG())
        rag.name, rag.description = "rag", "知识"
        inventory = Mock()
        inventory.query_with_fallback.return_value = inventory_result()
        llm = ScriptedLLM('[TOOL_CALL:inventory:{"query":"鲈鱼100"}]'
                          '[TOOL_CALL:rag:{"query":"新手怎么练习匀速收线？"}]', '依据工具回答。')
        app = LuyaAgent(Settings(), llm=llm, rag=rag, inventory=inventory)
        self.addCleanup(app.close)
        question = "推荐本店100元以内钓鲈鱼的饵，并解释新手怎么练习匀速收线"
        result = app.ask(question)
        self.assertEqual({e["tool"] for e in result["tool_calls"]}, {"inventory", "rag"})
        inventory.query_with_fallback.assert_called_once_with(question, 5, allow_budget_relaxation=True)
        rag.run.assert_called_once_with({"query": "新手怎么练习匀速收线？"})

    def test_expansion_failure_keeps_original_query_and_masks_secret(self):
        self.tool.options = RetrievalOptions(mqe=True)
        self.tool.expander = Mock()
        self.tool.expander.generate.side_effect = RuntimeError("unit-secret")
        result = self.tool.search_result("铅头钩挂底")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["retrieval"]["extra_llm_calls"], 1)
        self.assertTrue(result["retrieval"]["warnings"])
        self.assertNotIn("unit-secret", json.dumps(result))

    def test_api_token_usage_propagates_to_retrieval_diagnostics(self):
        settings = Settings(llm_base_url="https://unit.invalid/v1", llm_model="configured-model",
                            llm_api_key="unit-secret")
        llm = AgentChatLLM(settings, httpx.MockTransport(lambda r: httpx.Response(200, json={
            "choices": [{"message": {"content": '{"queries":["铅头钩的钩尖"]}'}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 12},
        })))
        self.tool.options = RetrievalOptions(mqe=True)
        self.tool.expansion_llm = llm
        self.tool.expander = QueryExpansion(llm)
        detail = self.tool.search_result("铅头钩挂底")["retrieval"]
        self.assertEqual(detail["extra_llm_calls"], 1)
        self.assertEqual(detail["extra_prompt_tokens"], 40)
        self.assertEqual(detail["extra_completion_tokens"], 12)
        self.assertTrue(detail["usage_complete"])

    def test_reranker_outage_is_error_not_empty_or_silent_fallback(self):
        self.ranker.score.side_effect = InventoryError("重排服务不可用")
        with self.assertRaisesRegex(InventoryError, "重排服务"):
            self.tool.search_result("铅头钩挂底")

    def test_bad_rerank_scores_rejected(self):
        for scores in [[0.5], [float("nan")] * 5, [2] * 5]:
            self.ranker.score.side_effect = None
            self.ranker.score.return_value = scores
            with self.subTest(scores=scores), self.assertRaises(InventoryError):
                self.tool.search_result("铅头钩挂底")

    def test_relevance_threshold_can_abstain(self):
        self.ranker.score.side_effect = lambda q, hits: [0.01] * len(hits)
        result = self.tool.search_result("铅头钩挂底")
        self.assertEqual(result["status"], "empty")
        self.assertTrue(result["retrieval"]["rerank_applied"])

    def test_sparse_cache_excludes_old_and_foreign_chunks(self):
        self.client.upsert(KNOWLEDGE_COLLECTION, [models.PointStruct(
            id=42, vector=[0, 1], payload={"rag_namespace": "foreign", "corpus_revision": self.tool.revision,
                                        "embedding_model": self.tool.settings.embedding_model,
                                        "source_file": "02_jighead_snags.md", "content": "量子计算机纠错"})])
        self.assertEqual(self.tool.search_result("量子计算机纠错")["status"], "empty")
        self.tool.revision = "old"
        with self.assertRaisesRegex(InventoryError, "已变更"):
            self.tool.search_result("铅头钩")

    def test_sparse_cache_reused_and_invalidated_by_index(self):
        self.tool.search_result("铅头钩挂底")
        cache = self.tool._lexical_cache
        self.tool.search_result("软饵挂底")
        self.assertIs(self.tool._lexical_cache, cache)
        with contextlib.redirect_stdout(io.StringIO()):
            self.tool.index_documents()
        self.assertIsNone(self.tool._lexical_cache)


class ExpansionAPITests(unittest.TestCase):
    def settings(self):
        return Settings(llm_base_url="https://unit.invalid/v1", llm_model="configured-model",
                        llm_api_key="unit-secret", llm_extra_body={"reasoning_effort": "low"})

    def test_expansion_json_mode_and_api_usage_accounting(self):
        def respond(request):
            self.body = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"queries":["钩尖挂底"]}'}}],
                                             "usage": {"prompt_tokens": 40, "completion_tokens": 12}})
        llm = AgentChatLLM(self.settings(), httpx.MockTransport(respond))
        self.assertEqual(QueryExpansion(llm).generate("铅头钩挂底", "mqe"), ["钩尖挂底"])
        self.assertEqual(self.body["response_format"], {"type": "json_object"})
        self.assertEqual(self.body["model"], "configured-model")
        self.assertEqual(llm.calls[0]["prompt_tokens"], 40)
        self.assertEqual(llm.calls[0]["completion_tokens"], 12)
        self.assertTrue(llm.calls[0]["ok"])

    def test_failure_counted_without_provider_secret(self):
        llm = AgentChatLLM(self.settings(), httpx.MockTransport(
            lambda r: httpx.Response(401, json={"error": "unit-secret"})))
        with self.assertRaises(InventoryError):
            llm.invoke([], json_mode=True)
        self.assertEqual(len(llm.calls), 1)
        self.assertFalse(llm.calls[0]["ok"])
        self.assertNotIn("unit-secret", json.dumps(llm.calls))


if __name__ == "__main__":
    unittest.main()
