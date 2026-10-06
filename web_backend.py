"""Phase 5 adapter: per-browser JSON context, per-callback SQLite connection."""

import copy
from dataclasses import replace
import html
import json
from threading import RLock
from urllib.parse import urlsplit

from knowledge_tool import SharedEmbedder
from luya_memory import ConversationMemory, DEFAULT_MEMORY_DB, PreferenceStore, validate_user_id
from memory_agent import MemoryLuyaAgent
from phase1_config import InventoryError, Settings
from rag_retrieval import LocalReranker, RetrievalOptions


def new_web_session():
    memory = ConversationMemory()
    return {"context": {"session_id": memory.session_id, "history": [], "slots": {}}, "messages": []}


class WebController:
    def __init__(self, settings=None, *, user_id="web-local", memory_path=DEFAULT_MEMORY_DB, agent_factory=None):
        self.settings = settings or Settings.from_env()
        self.user_id = validate_user_id(user_id)  # Server-owned, never a browser input.
        self.memory_path = memory_path
        self.agent_factory = agent_factory
        self.embedder = SharedEmbedder(self.settings.embedding_model)
        self.base_options = RetrievalOptions.from_env()
        self.reranker = LocalReranker(self.base_options.reranker_model)
        self.lock = RLock()

    def run(self, question, state, *, mode="hybrid", strict=False, allow_budget=True, mqe=False, hyde=False, record_chat=True):
        if not isinstance(question, str) or not question.strip() or len(question) > 4000:
            raise InventoryError("请输入 1–4000 字符的问题。")
        if any(type(flag) is not bool for flag in (strict, allow_budget, mqe, hyde, record_chat)):
            raise InventoryError("库存选项无效。")
        options = replace(self.base_options, mode=mode, rerank=True, mqe=mqe, hyde=hyde)
        state = copy.deepcopy(state or new_web_session())
        with self.lock:
            # Gradio callbacks can run in different worker threads. Never put a
            # live SQLite connection, Agent or model object into gr.State.
            store = PreferenceStore(self.memory_path)
            memory = ConversationMemory(self.user_id, store=store)
            memory.session_id = state["context"]["session_id"]
            memory.turns.extend(copy.deepcopy(state["context"]["history"]))
            memory.slots = copy.deepcopy(state["context"]["slots"])
            agent = None
            try:
                if self.agent_factory is not None:
                    agent = self.agent_factory(memory, options, strict, allow_budget)
                else:
                    agent = MemoryLuyaAgent(self.settings, user_id=self.user_id, memory=memory,
                                            rag_options=options, strict_inventory=strict,
                                            allow_budget_relaxation=allow_budget)
                    agent.shared_embedder = self.embedder
                    agent.rag.embedder = self.embedder
                    agent.rag.reranker = self.reranker
                result = agent.ask(question.strip())
                snapshot = memory.snapshot()
                state["context"] = {"session_id": snapshot["session_id"],
                                    "history": snapshot["history"], "slots": snapshot["session_slots"]}
            except InventoryError:
                raise
            except Exception as exc:
                # No third-party exception/provider body reaches the browser.
                raise InventoryError("暂时无法完成查询，请检查本地服务后重试。") from exc
            finally:
                if agent is not None:
                    agent.close()
                store.close()
        if question.strip() in {"/new", "/forget"} or question.strip().startswith("/forget "):
            state["messages"] = []
        elif record_chat:
            state["messages"] = (state["messages"] + [
                {"role": "user", "content": question.strip()},
                {"role": "assistant", "content": result["answer"]},
            ])[-40:]
        return state, result, snapshot

    def inspect(self, state=None):
        with self.lock:
            store = PreferenceStore(self.memory_path)
            try:
                context = (state or new_web_session())["context"]
                return {"user_id": self.user_id, "session_id": context["session_id"],
                        "preferences": store.load(self.user_id), "session_slots": context["slots"],
                        "history": context["history"]}
            finally:
                store.close()


def esc(value):
    return html.escape(str(value), quote=True)


def memory_html(snapshot):
    values = {**snapshot.get("preferences", {}), **snapshot.get("session_slots", {})}
    species = values.get("species") or "还没指定"
    budget = values.get("max_price")
    budget_label = f"¥{budget:g}以内" if budget is not None else "尚未设置 / 不限"
    water = values.get("water_type") or "还没指定"
    saved = len(snapshot.get("preferences", {}))
    return f'''<div class="memory-card"><div class="eyebrow">YOUR FIELD NOTES</div>
      <h3>这次出钓</h3><dl><div><dt>目标鱼种</dt><dd>{esc(species)}</dd></div>
      <div><dt>单件预算</dt><dd>{esc(budget_label)}</dd></div>
      <div><dt>水域类型</dt><dd>{esc(water)}</dd></div></dl>
      <div class="memory-foot"><span class="tiny-dot"></span> 已保留 {len(snapshot.get('history', []))} 轮上下文
      <span class="saved-count">{saved} 项长期偏好</span></div></div>'''


LURE_ICON = '''<svg viewBox="0 0 140 70" aria-hidden="true"><path d="M20 35C38 14 76 13 109 31L123 21L119 39L123 51L108 41C73 57 42 56 20 35Z" fill="currentColor" opacity=".18"/><path d="M20 35C38 14 76 13 109 31L123 21L119 39L123 51L108 41C73 57 42 56 20 35Z" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="37" cy="31" r="3" fill="currentColor"/><path d="M69 47v12c0 10 12 10 12 1m17-18v12c0 9 11 9 11 0" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'''


def products_html(result=None):
    inventories = (result or {}).get("inventory_results", [])
    products = [p for entry in inventories for p in entry.get("products", [])]
    if not products:
        if inventories:
            notice = " ".join(entry.get("reply", "") for entry in inventories)
            return '<div class="empty-result"><b>本轮没有库存候选</b><p>' + esc(notice) + '</p></div>'
        return '<div class="empty-result"><span class="eyebrow">CURATED, NOT GUESSED</span><b>合适的装备，来自真实筛选。</b><p>聊聊鱼种、预算和水况；推荐后，候选会出现在这里。</p></div>'
    cards = []
    for number, product in enumerate(products):
        deviations = product.get("deviations", [])
        warning = '<div class="product-warning">' + esc("；".join(deviations)) + ' · 需要你确认</div>' if deviations else ""
        tags = " · ".join(str(product.get(k, "")) for k in ("species", "water_type", "bait_type") if product.get(k))
        cards.append(f'''<article class="product-card {'alternative' if deviations else ''}">
          <div class="lure-art art-{number % 3}">{LURE_ICON}<span>{number + 1:02d}</span></div>
          <div class="product-copy"><div class="product-tags">{esc(tags)}</div><h3>{esc(product['name'])}</h3>
          <p>{esc(product.get('description', ''))}</p><div class="product-bottom"><strong>¥{float(product['price']):.2f}</strong>
          <span>{esc(product['stock'])} 件在库</span></div>{warning}</div></article>''')
    notices = " ".join(entry.get("reply", "") for entry in inventories)
    return f'<div class="result-note">{esc(notices)}</div><div class="product-grid">' + "".join(cards) + '</div>'


def sources_html(result=None):
    sources = (result or {}).get("sources", [])
    if not sources:
        return '<div class="source-empty">知识问答后，这里会展示本轮实际检索到的资料。库存推荐不代表知识来源。</div>'
    cards = []
    for source in sources:
        title = esc(source["title"])
        url = source.get("url")
        parsed = urlsplit(url) if isinstance(url, str) else None
        if parsed and parsed.scheme == "https" and parsed.netloc:
            title = f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{title}<span> ↗</span></a>'
        cards.append(f'<div class="source-item"><span class="source-index">{esc(source["citation"])}</span><div>{title}<small>{esc(source["source_file"])}</small></div></div>')
    return '<div class="source-list">' + "".join(cards) + '</div>'


def trace_text(result):
    calls = result.get("tool_calls", [])
    text = ["实际工具：" + (" → ".join(c["tool"] for c in calls) or "无（本地命令 / 普通对话）")]
    for call in calls:
        detail = call["result"].get("retrieval")
        if detail:
            text.append(f"检索：{detail['mode']} · 重排执行：{detail['rerank_applied']} · {detail['elapsed_ms']}ms")
            text.append(f"检索额外 LLM 调用：{detail['extra_llm_calls']}（不含路由及最终回答）")
            text.extend(detail.get("warnings", []))
    return "\n".join(text)
