"""Web session/thread boundaries, local commands and safe result rendering."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from memory_agent import MemoryLuyaAgent
from phase1_config import InventoryError, Settings
from test_phase3 import FakeRAG, ScriptedLLM, inventory_result
from web_backend import WebController, memory_html, new_web_session, products_html, sources_html


class WebTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="luya-web-test-")
        self.addCleanup(directory.cleanup)
        self.options = []

        def factory(memory, options, strict, allow_budget):
            self.options.append(options)
            inventory = Mock()
            result = inventory_result()
            result["slots"] = {"species": "鲈鱼", "max_price": 100, "water_type": None, "soft_query": "新手"}
            inventory.query_with_fallback.return_value = result
            inventory.query.return_value = result
            return MemoryLuyaAgent(Settings(), user_id=memory.user_id, memory=memory,
                llm=ScriptedLLM('[TOOL_CALL:inventory:{"query":"鲈鱼"}]', '候选已核验。'),
                inventory=inventory, rag=FakeRAG(), strict_inventory=strict,
                allow_budget_relaxation=allow_budget)

        self.controller = WebController(Settings(), memory_path=Path(directory.name) / "memory.sqlite3", agent_factory=factory)

    def test_new_browser_sessions_are_independent_and_input_not_mutated(self):
        a, b = new_web_session(), new_web_session()
        before = copy.deepcopy(a)
        result, _, snapshot = self.controller.run("推荐鲈鱼拟饵", a)
        self.assertEqual(a, before)
        self.assertNotEqual(a["context"]["session_id"], b["context"]["session_id"])
        self.assertEqual(result["context"]["slots"]["max_price"], 100)
        self.assertEqual(len(snapshot["history"]), 1)
        self.assertEqual(self.controller.inspect(b)["session_slots"], {})

    def test_preferences_persist_but_new_session_clears_only_chat(self):
        state, _, _ = self.controller.run('/remember {"species":"鲈鱼","max_price":80}', new_web_session(), record_chat=False)
        self.assertEqual(state["messages"], [])
        state, _, _ = self.controller.run("推荐拟饵", state)
        old_id = state["context"]["session_id"]
        state, _, snapshot = self.controller.run("/new", state)
        self.assertEqual(state["messages"], [])
        self.assertEqual(snapshot["history"], [])
        self.assertEqual(snapshot["preferences"]["max_price"], 80)
        self.assertNotEqual(state["context"]["session_id"], old_id)
        self.assertEqual(self.controller.inspect(new_web_session())["preferences"]["species"], "鲈鱼")

    def test_form_save_never_drops_old_chat_at_limit(self):
        state = new_web_session()
        state["messages"] = [{"role":"user", "content":str(i)} for i in range(40)]
        saved, _, _ = self.controller.run('/remember {"max_price":50}', state, record_chat=False)
        self.assertEqual(saved["messages"], state["messages"])

    def test_forget_clears_profile_and_short_context(self):
        state, _, _ = self.controller.run('/remember {"species":"黑鱼"}', new_web_session())
        state, _, snapshot = self.controller.run("/forget", state)
        self.assertEqual(snapshot["preferences"], {})
        self.assertEqual(state["messages"], [])

    def test_callback_threads_do_not_share_sqlite_handles(self):
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda _: self.controller.run("/help", new_web_session()), range(6)))
        self.assertEqual(len({r[0]["context"]["session_id"] for r in results}), 6)
        self.assertTrue(all(r[1]["memory_command"] == "/help" for r in results))

    def test_invalid_input_and_options_fail_before_agent(self):
        for question, options in [("", {}), ("x" * 4001, {}), ("hello", {"mode":"basic", "hyde":True}),
                                  ("hello", {"mqe":"false"}), ("hello", {"strict":1})]:
            with self.subTest(options=options), self.assertRaises(InventoryError):
                self.controller.run(question, new_web_session(), **options)
        self.assertEqual(self.options, [])

    def test_options_preserve_server_thresholds(self):
        from dataclasses import replace
        self.controller.base_options = replace(self.controller.base_options, candidate_k=30, min_rerank_score=.4)
        self.controller.run("/help", new_web_session(), mode="basic")
        self.assertEqual(self.options[-1].candidate_k, 30)
        self.assertEqual(self.options[-1].min_rerank_score, .4)

    def test_unexpected_provider_exception_masked(self):
        agent = Mock()
        agent.ask.side_effect = RuntimeError("secret provider response")
        self.controller.agent_factory = lambda *args: agent
        with self.assertRaises(InventoryError) as captured:
            self.controller.run("hello", new_web_session())
        self.assertNotIn("secret", str(captured.exception))
        agent.close.assert_called_once()


class RenderTests(unittest.TestCase):
    def test_product_markup_escaped_and_deviations_visible(self):
        result = inventory_result("alternatives")
        result["products"][0]["name"] = '<script>alert("x")</script>'
        rendered = products_html({"inventory_results":[result]})
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("48.00", rendered)
        self.assertIn("超过原预算 8 元", rendered)
        self.assertIn("需要你确认", rendered)

    def test_sources_only_link_https_and_escape_labels(self):
        rendered = sources_html({"sources":[{"title":"<img>", "citation":"K1", "source_file":"a.md", "url":"javascript:alert(1)"}]})
        self.assertNotIn("href=", rendered)
        self.assertNotIn("<img>", rendered)
        rendered = sources_html({"sources":[{"title":"官方", "citation":"K1", "source_file":"a.md", "url":"https://example.org/?a=1&b=2"}]})
        self.assertIn("noopener noreferrer", rendered)
        self.assertIn("&amp;", rendered)

    def test_current_null_overrides_saved_budget_and_missing_fields_merge(self):
        rendered = memory_html({"preferences":{"species":"鲈鱼","max_price":100}, "session_slots":{"max_price":None}})
        self.assertIn("鲈鱼", rendered)
        self.assertIn("尚未设置 / 不限", rendered)
        self.assertNotIn("¥100", rendered)


if __name__ == "__main__":
    unittest.main()
