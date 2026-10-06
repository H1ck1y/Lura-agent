"""Phase 4 memory persistence, scope, priority and framework integration."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import httpx

from luya_memory import (ConversationMemory, MemoryAwareLLM, MemorySlotExtractor,
                         PreferenceStore, validate_preferences)
from memory_agent import MemoryLuyaAgent
from phase1_config import InventoryError, Settings
from test_phase3 import FakeRAG, ScriptedLLM, inventory_result


def slots(**changes):
    return {"species": "鲈鱼", "max_price": 100.0, "water_type": "清水",
            "soft_query": "新手友好，不易挂底", **changes}


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="luya-phase4-test-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "memory.sqlite3"
        self.store = PreferenceStore(self.path)
        self.addCleanup(self.store.close)

    def test_explicit_preferences_persist_across_restart_and_are_user_isolated(self):
        self.store.save("alice", {"species": "鲈鱼", "max_price": 100})
        reopened = PreferenceStore(self.path)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.load("alice")["species"], "鲈鱼")
        self.assertEqual(reopened.load("bob"), {})
        reopened.save("bob", {"species": "黑鱼"})
        self.assertEqual(self.store.load("alice")["species"], "鲈鱼")

    def test_updates_merge_and_delete_are_scoped(self):
        self.store.save("alice", {"species": "鲈鱼", "max_price": 100})
        self.store.save("alice", {"max_price": 50})
        self.store.save("bob", {"species": "黑鱼"})
        self.assertEqual(self.store.load("alice"), {"species": "鲈鱼", "max_price": 50})
        self.store.forget("alice", ["max_price"])
        self.assertEqual(self.store.load("alice"), {"species": "鲈鱼"})
        self.store.forget("alice")
        self.assertEqual(self.store.load("alice"), {})
        self.assertEqual(self.store.load("bob"), {"species": "黑鱼"})

    def test_invalid_or_secret_preferences_rejected_without_changes(self):
        for data in [{}, {"notes": "x"}, {"max_price": -1}, {"max_price": "100"},
                     {"max_price": True}, {"water_type": "淡水"}, {"species": None},
                     {"species": "x" * 41}, {"soft_query": "sk-test_secret123456789"},
                     {"soft_query": "密码=private"}]:
            with self.subTest(data=data), self.assertRaises(InventoryError):
                self.store.save("alice", data)
        self.assertEqual(self.store.load("alice"), {})

    def test_unsafe_user_identifiers_rejected(self):
        for user in ["", "../bob", "a' OR 1=1", "x" * 65, None]:
            with self.subTest(user=user), self.assertRaises(InventoryError):
                self.store.load(user)

    def test_corrupted_database_not_silently_overwritten(self):
        self.store.db.execute("INSERT INTO luya_preferences VALUES (?,?)", ("alice", "bad-json"))
        self.store.db.commit()
        with self.assertRaises(InventoryError):
            self.store.save("alice", {"species": "鲈鱼"})
        self.assertEqual(self.store.db.execute("SELECT data FROM luya_preferences").fetchone()[0], "bad-json")


class ContextTests(unittest.TestCase):
    def result(self, **changes):
        return {"status": "ok", "answer": "回答", "tool_calls": [{"tool": "inventory"}],
                "inventory_results": [{"slots": slots(), "effective_slots": slots(max_price=120)}], **changes}

    def test_bounded_context_and_original_budget_not_relaxed_budget(self):
        memory = ConversationMemory(max_turns=2)
        for i in range(3):
            memory.commit(str(i), self.result())
        self.assertEqual([t["question"] for t in memory.turns], ["1", "2"])
        self.assertEqual(memory.slots["max_price"], 100)

    def test_failed_turn_does_not_change_memory(self):
        memory = ConversationMemory()
        memory.commit("old", self.result())
        before = memory.snapshot()
        memory.commit("new", self.result(status="partial", inventory_results=[{"slots": slots(max_price=50)}]))
        self.assertEqual(memory.snapshot(), before)

    def test_clear_constraints_override_saved_defaults_including_zero_budget(self):
        store = Mock()
        store.load.return_value = {"max_price": 100, "species": "鲈鱼"}
        memory = ConversationMemory(store=store)
        memory.commit("不限制预算", self.result(inventory_results=[{"slots": slots(max_price=None)}]))
        self.assertIsNone(memory.snapshot()["defaults"]["max_price"])
        memory.commit("预算为零", self.result(inventory_results=[{"slots": slots(max_price=0)}]))
        self.assertEqual(memory.snapshot()["defaults"]["max_price"], 0)

    def test_session_context_not_shared_between_users_or_instances(self):
        alice, bob = ConversationMemory("alice"), ConversationMemory("bob")
        alice.commit("alice budget", self.result())
        self.assertEqual(bob.snapshot()["history"], [])
        self.assertNotEqual(alice.session_id, bob.session_id)

    def test_new_session_clears_context_but_retains_preferences(self):
        store = Mock()
        store.load.return_value = {"species": "鲈鱼"}
        memory = ConversationMemory(store=store)
        memory.commit("old", self.result())
        old_id = memory.session_id
        memory.new_session()
        self.assertEqual(memory.snapshot()["history"], [])
        self.assertEqual(memory.snapshot()["defaults"], {"species": "鲈鱼"})
        self.assertNotEqual(old_id, memory.session_id)

    def test_snapshot_defensive_copy_and_best_effort_secret_redaction(self):
        memory = ConversationMemory()
        memory.commit("API_KEY=private sk-testsecret12345", self.result(answer="密码：abc"))
        snapshot = memory.snapshot()
        self.assertNotIn("private", json.dumps(snapshot))
        self.assertNotIn("sk-testsecret12345", json.dumps(snapshot))
        snapshot["defaults"]["max_price"] = 1
        snapshot["history"].clear()
        self.assertEqual(memory.slots["max_price"], 100)
        self.assertEqual(len(memory.turns), 1)

    def test_bad_tool_slots_not_committed(self):
        memory = ConversationMemory()
        with self.assertRaises(InventoryError):
            memory.commit("bad", self.result(inventory_results=[{"slots": {"max_price": -1}}]))
        self.assertEqual(memory.slots, {})
        self.assertEqual(len(memory.turns), 0)


class ModelContextTests(unittest.TestCase):
    def test_memory_injected_as_data_without_mutating_original_messages(self):
        memory = ConversationMemory()
        memory.turns.append({"question": "历史数据", "answer": "旧价格", "tools": []})
        llm = ScriptedLLM("回答", '{"queries":[]}')
        wrapped = MemoryAwareLLM(llm, memory)
        messages = [{"role": "system", "content": "规则"}, {"role": "user", "content": "当前问题"}]
        before = copy.deepcopy(messages)
        wrapped.invoke(messages)
        self.assertEqual(messages, before)
        self.assertEqual(llm.requests[0][0], messages[0])
        self.assertEqual(llm.requests[0][1]["role"], "user")
        self.assertEqual(llm.requests[0][-1]["content"], "当前问题")
        wrapped.invoke(messages, json_mode=True)
        self.assertEqual(llm.requests[1], messages)

    def test_context_slot_extractor_preserves_current_input_and_validates_response(self):
        memory = ConversationMemory()
        memory.slots = slots()
        captured = []
        def respond(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(slots(max_price=50))}}]})
        settings = Settings(llm_base_url="https://unit.invalid/v1", llm_model="same-model", llm_api_key="unit-secret")
        extractor = MemorySlotExtractor(settings, memory, httpx.MockTransport(respond))
        output = extractor.extract("预算改成50", {"species": [], "aliases": {}})
        self.assertEqual(output.max_price, 50)
        self.assertEqual(captured[0]["messages"][-1]["content"], "预算改成50")
        self.assertIn('"max_price": 100.0', captured[0]["messages"][1]["content"])
        self.assertIn("当前问题的明确条件", captured[0]["messages"][0]["content"])


class MemoryAgentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="luya-phase4-agent-")
        self.addCleanup(self.directory.cleanup)
        self.store = PreferenceStore(Path(self.directory.name) / "memory.sqlite3")
        self.addCleanup(self.store.close)
        self.memory = ConversationMemory("alice", store=self.store)

    def app(self, *responses, result=None):
        llm = ScriptedLLM(*responses)
        inventory = Mock()
        inventory.query_with_fallback.return_value = result or {**inventory_result(), "slots": slots()}
        app = MemoryLuyaAgent(Settings(), memory=self.memory, llm=llm, inventory=inventory, rag=FakeRAG())
        self.addCleanup(app.close)
        return app, llm, inventory

    def test_multiturn_context_reaches_model_and_inventory_is_fresh_each_time(self):
        app, llm, inventory = self.app('[TOOL_CALL:inventory:query=鲈鱼100]', '第一轮回答',
                                       '[TOOL_CALL:inventory:query=更便宜]', '第二轮回答')
        app.ask("钓鲈鱼预算100，请推荐")
        result = app.ask("再便宜一点")
        self.assertEqual(inventory.query_with_fallback.call_count, 2)
        self.assertEqual(inventory.query_with_fallback.call_args.args[0], "再便宜一点")
        self.assertIn("第一轮回答", json.dumps(llm.requests[2], ensure_ascii=False))
        self.assertEqual(result["memory"]["defaults"]["max_price"], 100)
        self.assertEqual(app.agent._history, [])
        self.assertEqual(self.store.load("alice"), {})

    def test_knowledge_turns_enter_bounded_context(self):
        app, llm, inventory = self.app('[TOOL_CALL:rag:query=铅头钩挂底]', '露钩尖风险。[K1]')
        app.ask("铅头钩为什么挂底")
        self.assertEqual(self.memory.turns[0]["tools"], ["rag"])
        inventory.query_with_fallback.assert_not_called()

    def test_management_commands_never_invoke_model_or_inventory(self):
        app, llm, inventory = self.app()
        app.ask('/remember {"species":"鲈鱼","max_price":100}')
        self.assertEqual(self.store.load("alice")["species"], "鲈鱼")
        self.assertIn("鲈鱼", app.ask("/memory")["answer"])
        app.ask("/new")
        self.assertEqual(self.memory.snapshot()["defaults"]["max_price"], 100)
        app.ask("/forget max_price")
        self.assertEqual(self.store.load("alice"), {"species": "鲈鱼"})
        app.ask("/forget")
        self.assertEqual(self.store.load("alice"), {})
        self.assertEqual(llm.requests, [])
        inventory.query_with_fallback.assert_not_called()

    def test_remember_overrides_stale_session_field(self):
        app, _, _ = self.app()
        self.memory.slots = slots(max_price=80)
        app.ask('/remember {"max_price":50}')
        self.assertEqual(self.memory.snapshot()["defaults"]["max_price"], 50)

    def test_forget_also_clears_session_and_history(self):
        app, _, _ = self.app()
        self.store.save("alice", {"species": "鲈鱼"})
        self.memory.slots = slots()
        self.memory.turns.append({"question": "钓鲈鱼", "answer": "x", "tools": []})
        app.ask("/forget species")
        self.assertEqual(self.memory.snapshot()["defaults"], {})
        self.assertEqual(list(self.memory.turns), [])

    def test_invalid_commands_do_not_call_model_or_mutate(self):
        app, llm, _ = self.app()
        for command in ['/remember {bad}', '/remember {"notes":"bad"}', '/forget unknown', '/new extra', '/clear',
                        '/remember sk-testsecret123456789']:
            with self.subTest(command=command), self.assertRaises(InventoryError):
                app.ask(command)
        self.assertEqual(llm.requests, [])
        self.assertEqual(self.store.load("alice"), {})

    def test_failed_model_turn_not_committed(self):
        app, _, _ = self.app('[TOOL_CALL:unknown:query=x]')
        with self.assertRaises(InventoryError):
            app.ask("bad")
        self.assertEqual(list(self.memory.turns), [])

    def test_natural_language_remember_saves_only_explicit_supported_fields(self):
        from unittest.mock import patch
        from llm_slots import Slots
        app, _, _ = self.app()
        with patch("memory_agent.SlotExtractor") as factory:
            factory.return_value.extract.return_value = Slots.model_validate(
                slots(max_price=None, water_type=None))
            app.ask("/remember 我是新手，通常钓鲈鱼")
            factory.return_value.extract.assert_called_once_with("我是新手，通常钓鲈鱼", {"species": [], "aliases": {}})
        data = self.store.load("alice")
        self.assertNotIn("max_price", data)
        self.assertNotIn("water_type", data)
        self.assertEqual(data["species"], "鲈鱼")

    def test_invalid_user_id_rejected_before_memory_file_creation(self):
        path = Path(self.directory.name) / "must-not-create.sqlite3"
        with self.assertRaises(InventoryError):
            MemoryLuyaAgent(Settings(), user_id="../alice", memory_path=path, rag=FakeRAG())
        self.assertFalse(path.exists())

    def test_new_and_forget_clear_framework_tool_journal(self):
        app, _, _ = self.app()
        app.journal.append({"tool": "inventory", "result": {"slots": slots()}})
        app.ask("/new")
        self.assertEqual(app.journal, [])
        app.journal.append({"tool": "inventory", "result": {"slots": slots()}})
        app.ask("/forget")
        self.assertEqual(app.journal, [])


if __name__ == "__main__":
    unittest.main()
