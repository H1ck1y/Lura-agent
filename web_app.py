"""Phase 5 Gradio workbench. Local text-only UI over the existing agent."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys

import gradio as gr

from phase1_config import InventoryError
from web_backend import WebController, memory_html, new_web_session, products_html, sources_html, trace_text


ROOT = Path(__file__).resolve().parent
QUESTIONS = [
    ("01  新手选饵", "我是新手，钓鲈鱼，预算100元以内，最好不容易挂底，推荐本店商品"),
    ("02  清浑水选色", "清水和浑水怎么选择拟饵颜色？只讲知识"),
    ("03  挂底的原因", "铅头钩配软饵为什么容易挂底？只讲原理"),
    ("04  选购＋技巧", "推荐本店100元以内钓鲈鱼的饵，并解释新手怎么练习匀速收线"),
]
BRAND = '''<div class="brand"><div class="brand-mark"><svg viewBox="0 0 40 40" aria-hidden="true"><path d="M7 21c7-11 16-11 22-1l5-5v12l-5-5c-7 8-16 8-22-1Z" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="12" cy="20" r="1.5" fill="currentColor"/><path d="M8 30c6-3 12 3 20 0" stroke="currentColor" fill="none" stroke-width="1.3"/></svg></div><div><strong>野水</strong><small>WILDWATER · LUYA</small></div></div>'''
HERO = '''<div class="topline"><span>出钓工作台 / FIELD DESK</span><span class="local-badge"><i class="tiny-dot"></i> 本地运行 · 文字咨询</span></div>
<section class="hero"><div class="eyebrow">A BETTER CAST STARTS HERE</div><h1>找到你的下一口。</h1><p>从一款合适的拟饵，到一次更从容的抛投。<br>聊聊你的出钓计划，让选择有据可依。</p>
<svg class="landscape" viewBox="0 0 440 220" preserveAspectRatio="xMidYMid slice" aria-hidden="true"><circle cx="312" cy="47" r="22" fill="#d4dcc5"/><path d="m70 152 65-75 61 68 41-44 49 51 37-70 64 70 56-38 52 75H35Z" fill="#d9e2cd"/><path d="m122 175 64-61 55 45 46-32 80 51 66-28 52 45H25Z" fill="#c9d6bc"/><path d="M-20 191c95-23 128 20 224 0s157-2 254-8v43H-20" fill="#afc3a2"/><path d="M55 207c86-24 116 15 236-3m-183-28 57-1m169 25 42-3" fill="none" stroke="#eff3e8" stroke-width="2" stroke-linecap="round"/><path d="M290 167v-27m0 4 8 3m-8 10-5 12m5-12 5 12" stroke="#73916a" stroke-width="2" fill="none"/><circle cx="290" cy="136" r="3" fill="#73916a"/><path d="m298 147 24-40c19 16 30 42 31 67" fill="none" stroke="#73916a" stroke-width="1"/></svg></section>'''
WELCOME = '''<div class="chat-welcome"><div class="welcome-icon">⌁</div><b>今天，打算在哪里抛一竿？</b><p>告诉我目标鱼、预算和水况，或问一个钓法问题。<br>也可以从上面的灵感问题开始。</p></div>'''


def build_app(controller=None):
    controller = controller or WebController()
    with gr.Blocks(title="野水 · 路亚出钓工作台", analytics_enabled=False, fill_width=True) as demo:
        session = gr.State(value=new_web_session, time_to_live=3600)
        with gr.Row(elem_id="workbench"):
            with gr.Column(scale=0, min_width=234, elem_id="side-rail"):
                gr.HTML(BRAND)
                gr.HTML('<div class="rail-nav"><span>⌁</span> 咨询工作台 <span>↗</span></div>')
                new_button = gr.Button("＋  开启新对话", elem_id="new-session")
                memory_view = gr.HTML(memory_html({}), elem_id="memory-view")
                with gr.Accordion("长期偏好", open=False, elem_id="profile-settings"):
                    gr.Markdown("仅点保存才会长期记住。留空不修改已有字段。")
                    species = gr.Textbox(label="常钓鱼种", placeholder="例如：鲈鱼", max_lines=1)
                    budget = gr.Number(label="单件预算上限（元）", value=None, minimum=0, precision=2)
                    water = gr.Dropdown(["未设置", "清水", "浑水", "咸水", "通用"], value="未设置", label="常用水域")
                    soft = gr.Textbox(label="软性偏好", placeholder="例如：新手友好、少挂底", lines=2, max_lines=3)
                    save = gr.Button("保存到本地偏好", variant="primary")
                    confirm_forget = gr.Checkbox(label="确认清除全部长期偏好与当前对话", value=False)
                    forget = gr.Button("清除我的偏好", variant="secondary")
                gr.HTML('<div class="rail-note">只做咨询与推荐，不下单、不付款。<br>同一服务使用一个本地档案；不同标签页的短期对话互不共享。</div>')
            with gr.Column(scale=1, min_width=400, elem_id="main-area"):
                gr.HTML(HERO)
                gr.HTML('<div class="section-label"><h2>从这里开始</h2><small>点击填入 · 不会自动发送</small></div>')
                with gr.Row(elem_id="question-chips"):
                    chips = [gr.Button(title, min_width=120) for title, _ in QUESTIONS]
                with gr.Column(elem_id="conversation-shell"):
                    gr.HTML('<div class="conversation-head"><strong>你的路亚顾问</strong><span>库存核验 / 资料可追溯</span></div>')
                    chat = gr.Chatbot(value=[], height=345, show_label=False, container=False,
                                      layout="bubble", placeholder=WELCOME, buttons=["copy"],
                                      sanitize_html=True, allow_tags=False, allow_file_downloads=False,
                                      elem_id="conversation")
                    with gr.Row(elem_id="composer-row"):
                        question = gr.Textbox(placeholder="例如：新手钓鲈鱼，预算100元以内，最好少挂底……",
                                              lines=2, max_lines=5, show_label=False, container=False,
                                              scale=8, elem_id="question-input")
                        send = gr.Button("发送  ↗", variant="primary", scale=1, min_width=90, elem_id="send")
                    gr.HTML('<div class="composer-foot"><span>Enter 发送 · Shift + Enter 换行</span><span>不易挂底 ≠ 不会挂底 · 不保证钓获</span></div>')
                    status = gr.Markdown("准备好了。第一次查询可能需要加载本地模型。", elem_id="ui-status")
                with gr.Column(elem_id="result-area"):
                    gr.HTML('<div class="section-label"><h2>本轮装备候选</h2><small>库存工具实际返回 · 无购买功能</small></div>')
                    products = gr.HTML(products_html(), elem_id="product-view")
                    gr.HTML('<div class="section-label"><h2>本轮知识来源</h2><small>不是模型凭空生成的出处</small></div>')
                    sources = gr.HTML(sources_html(), elem_id="sources-view")
                with gr.Accordion("检索设置与调用记录", open=False, elem_id="advanced-settings"):
                    with gr.Row():
                        mode = gr.Radio(["hybrid", "basic"], value="hybrid", label="知识检索（hybrid 默认含重排）")
                        strict = gr.Checkbox(value=False, label="库存只接受严格匹配")
                        allow_budget = gr.Checkbox(value=True, label="允许展示最多超预算20%的替代候选")
                    with gr.Row():
                        mqe = gr.Checkbox(value=False, label="MQE 多查询扩展（额外 API 调用）")
                        hyde = gr.Checkbox(value=False, label="HyDE 假设文本检索（额外 API 调用）")
                    gr.Markdown("basic 模式不支持 MQE/HyDE。替代候选会保留差异，不能当作严格匹配。")
                    trace = gr.Textbox(label="实际执行记录（不展示模型思考）", value="尚未执行查询。", interactive=False, lines=3, elem_id="trace-output")
                gr.HTML('<div class="page-foot"><span>野水 WILDWATER · 让每次选择，更有依据。</span><span>演示库存与入门资料 · 本地档案非登录账号</span></div>')

        outputs = [chat, session, question, memory_view, products, sources, status, trace, send]
        forms = [species, budget, water, soft]

        def view(state, result, snapshot, notice=None):
            return (state["messages"], state, gr.update(value="", interactive=True), memory_html(snapshot),
                    products_html(result), sources_html(result), notice or "已完成。本轮候选与资料见下方。",
                    trace_text(result), gr.update(interactive=True))

        def submit(text, state, selected_mode, strict_match, budget_relax, enable_mqe, enable_hyde):
            state = state or new_web_session()
            if not isinstance(text, str) or not text.strip() or len(text) > 4000:
                yield (gr.skip(), state, gr.update(interactive=True), gr.skip(), gr.skip(), gr.skip(),
                       "请输入 1–4000 字符的问题。", gr.skip(), gr.update(interactive=True))
                return
            pending = (state["messages"] + [{"role": "user", "content": text.strip()}])[-40:]
            yield (pending, state, gr.update(value="", interactive=False), gr.skip(), gr.skip(), gr.skip(),
                   "正在理解问题并查询，请稍候……", gr.skip(), gr.update(interactive=False))
            try:
                state, result, snapshot = controller.run(text, state, mode=selected_mode, strict=strict_match,
                                                         allow_budget=budget_relax, mqe=enable_mqe, hyde=enable_hyde)
                notice = "本轮部分服务失败，请查看回答中的提示。" if result["status"] == "partial" else None
                if result.get("memory_command"):
                    notice = result["answer"] if result["memory_command"] != "/memory" else "记忆详情已展示在对话中。"
                yield view(state, result, snapshot, notice)
            except InventoryError as exc:
                # Preserve last confirmed memory and previous result panels.
                failed = copy.deepcopy(state)
                failed["messages"] = (pending + [{"role": "assistant", "content": "本次未完成：" + str(exc)}])[-40:]
                yield (failed["messages"], failed, gr.update(value=text, interactive=True), gr.skip(), gr.skip(), gr.skip(),
                       "本次没有更新会话条件。检查提示后可重试。", "本次请求未完成。", gr.update(interactive=True))

        def manage(command, state):
            try:
                state, result, snapshot = controller.run(command, state, record_chat=False)
                return view(state, result, snapshot, result["answer"])
            except InventoryError as exc:
                return (gr.skip(), state, gr.skip(), gr.skip(), gr.skip(), gr.skip(), str(exc), gr.skip(), gr.update(interactive=True))

        def save_preferences(target, ceiling, water_type, preferences, state):
            patch = {}
            if target and target.strip():
                patch["species"] = target.strip()
            if ceiling is not None:
                patch["max_price"] = ceiling
            if water_type != "未设置":
                patch["water_type"] = water_type
            if preferences and preferences.strip():
                patch["soft_query"] = preferences.strip()
            if not patch:
                return (gr.skip(), state, gr.skip(), gr.skip(), gr.skip(), gr.skip(), "请至少填写一项偏好。", gr.skip(), gr.skip())
            return manage("/remember " + json.dumps(patch, ensure_ascii=False), state)

        def forget_preferences(confirmed, state):
            if not confirmed:
                return (gr.skip(), state, gr.skip(), gr.skip(), gr.skip(), gr.skip(), "请先勾选确认，再清除偏好。", gr.skip(), gr.skip(),
                        gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())
            result = manage("/forget", state)
            # On failure, do not visually claim deletion by clearing the form.
            if result[0] == gr.skip():
                return result + (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())
            return result + ("", None, "未设置", "", False)

        def load(state):
            state = state or new_web_session()
            try:
                snapshot = controller.inspect(state)
                values = snapshot["preferences"]
                return state, memory_html(snapshot), values.get("species", ""), values.get("max_price"), values.get("water_type", "未设置"), values.get("soft_query", "")
            except InventoryError:
                return state, memory_html({}), "", None, "未设置", ""

        event_options = dict(concurrency_id="agent-workbench", concurrency_limit=1, api_visibility="private", show_progress="hidden")
        inputs = [question, session, mode, strict, allow_budget, mqe, hyde]
        send.click(submit, inputs, outputs, **event_options)
        question.submit(submit, inputs, outputs, **event_options)
        for chip, (_, sample) in zip(chips, QUESTIONS):
            chip.click(lambda text=sample: text, None, question, queue=False, api_visibility="private")
        new_button.click(lambda state: manage("/new", state), session, outputs, **event_options)
        # Chatbot's built-in trash control must clear memory, not only bubbles.
        chat.clear(lambda state: manage("/new", state), session, outputs, **event_options)
        save.click(save_preferences, [species, budget, water, soft, session], outputs, **event_options)
        forget.click(forget_preferences, [confirm_forget, session], outputs + forms + [confirm_forget], **event_options)
        demo.load(load, session, [session, memory_view] + forms, **event_options)

    demo.queue(max_size=8, default_concurrency_limit=1)
    return demo


def launch_options(port=7860):
    theme = gr.themes.Soft(primary_hue="green", neutral_hue="stone",
                           font=["Segoe UI", "Microsoft YaHei", "sans-serif"],
                           font_mono=["Consolas", "monospace"]).set(body_background_fill="#f7f7f2")
    return {"server_name": "127.0.0.1", "server_port": port, "share": False,
            "theme": theme, "css": (ROOT / "assets" / "web.css").read_text(encoding="utf-8"),
            "show_error": False, "footer_links": [], "run_history": False,
            "blocked_paths": [str(ROOT)], "state_session_capacity": 64, "max_threads": 8,
            "strict_cors": True}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    # Gradio's startup self-check uses HTTPX environment proxy settings.
    # Bypass only loopback; keep the user's proxy for external model requests.
    for name in ("NO_PROXY", "no_proxy"):
        entries = [v.strip() for v in os.environ.get(name, "").split(",") if v.strip()]
        os.environ[name] = ",".join(dict.fromkeys(entries + ["127.0.0.1", "localhost", "::1"]))
    parser = argparse.ArgumentParser(description="Phase 5 野水 · 路亚出钓工作台（本机文字界面）")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--user-id", default="web-local", help="服务绑定的本地偏好档案，不是账号认证")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("端口必须为 1024–65535。")
    build_app(WebController(user_id=args.user_id)).launch(**launch_options(args.port))


if __name__ == "__main__":
    try:
        main()
    except InventoryError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
