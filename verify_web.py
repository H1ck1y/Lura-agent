"""Real browser smoke test. Temporary preferences; no API calls unless requested."""
import argparse
import os
from pathlib import Path
import tempfile

from playwright.sync_api import sync_playwright, expect

from web_app import ROOT, build_app, launch_options
from web_backend import WebController


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true", help="Also send real inventory, follow-up and RAG questions (billable)")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",localhost,127.0.0.1,::1"
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    preview = ROOT / ".local" / "previews"
    preview.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="luya-web-smoke-") as directory:
        controller = WebController(memory_path=Path(directory) / "memory.sqlite3")
        demo = build_app(controller)
        demo.launch(**launch_options(args.port), prevent_thread_lock=True, quiet=True)
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel="msedge", headless=True)
                page = browser.new_page(viewport={"width":1440, "height":1100})
                errors = []
                page.on("pageerror", lambda exc: errors.append(str(exc)))
                page.goto(f"http://127.0.0.1:{args.port}", wait_until="domcontentloaded")
                expect(page.get_by_role("heading", name="找到你的下一口。")).to_be_visible()
                text = page.locator("#question-input textarea")
                send = page.locator("#send")

                def submit(question):
                    text.fill(question)
                    send.click()
                    # Wait for the first server update, not the previously enabled input.
                    expect(text).to_have_value("", timeout=30000)
                    expect(text).to_be_enabled(timeout=240000)
                    expect(page.locator("#ui-status")).not_to_contain_text("正在理解", timeout=240000)

                page.get_by_role("button", name="01 新手选饵").click()
                expect(text).to_have_value("我是新手，钓鲈鱼，预算100元以内，最好不容易挂底，推荐本店商品")
                expect(page.locator("#conversation")).not_to_contain_text("/help")
                submit("/help")
                expect(page.locator("#conversation")).to_contain_text("/remember", timeout=20000)
                page.locator("#conversation").get_by_role("button", name="清空对话").click()
                expect(page.locator("#ui-status")).to_contain_text("已清空当前会话", timeout=20000)
                expect(page.locator("#conversation")).not_to_contain_text("/remember")
                expect(page.locator("#conversation")).not_to_contain_text("/remember")

                page.get_by_text("长期偏好", exact=True).click()
                page.get_by_label("常钓鱼种", exact=True).fill("鲈鱼")
                page.get_by_label("单件预算上限（元）", exact=True).fill("100")
                page.get_by_role("button", name="保存到本地偏好").click()
                expect(page.locator("#ui-status")).to_contain_text("已明确保存长期偏好", timeout=20000)
                expect(page.locator("#memory-view")).to_contain_text("¥100以内")
                expect(page.locator("#conversation")).not_to_contain_text("/remember")
                page.get_by_role("button", name="清除我的偏好").click()
                expect(page.locator("#ui-status")).to_contain_text("请先勾选确认", timeout=20000)
                page.get_by_label("确认清除全部长期偏好与当前对话", exact=True).check()
                page.get_by_role("button", name="清除我的偏好").click()
                expect(page.locator("#ui-status")).to_contain_text("已删除指定长期偏好", timeout=20000)
                expect(page.locator("#memory-view")).to_contain_text("0 项长期偏好")
                page.get_by_text("长期偏好", exact=True).click()

                if args.with_llm:
                    submit("我是新手，钓鲈鱼，预算100元以内，最好不容易挂底，推荐本店商品")
                    expect(page.locator(".product-card").first).to_be_visible(timeout=20000)
                    expect(page.locator("#memory-view")).to_contain_text("¥100以内")
                    page.screenshot(path=str(preview / "phase5-inventory.png"), full_page=True)
                    submit("预算改成50元以内")
                    expect(page.locator("#memory-view")).to_contain_text("¥50以内", timeout=20000)
                    expect(page.locator(".product-card").first).to_be_visible()
                    submit("铅头钩配软饵为什么容易挂底？只讲原理")
                    expect(page.locator(".source-item").first).to_be_visible(timeout=20000)
                    page.screenshot(path=str(preview / "phase5-rag.png"), full_page=True)
                    print("PASS: real inventory, budget follow-up and cited knowledge")

                page.get_by_role("button", name="＋ 开启新对话").click()
                expect(page.locator("#ui-status")).to_contain_text("已清空当前会话", timeout=20000)
                page.screenshot(path=str(preview / "phase5-desktop.png"), full_page=True)
                page.set_viewport_size({"width":390, "height":844})
                expect(page.locator("#question-input textarea")).to_be_visible()
                page.screenshot(path=str(preview / "phase5-mobile.png"), full_page=True)
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "Mobile horizontal overflow"
                assert page.locator("#main-area").bounding_box()["width"] >= 340, "Mobile content is unnecessarily narrow"
                assert not errors, "Browser JavaScript errors: " + repr(errors)
                browser.close()
                print("PASS: suggestions, local commands, preferences, confirmation, desktop/mobile, JavaScript")
        finally:
            # A failed assertion must not race SQLite cleanup with an active callback.
            with controller.lock:
                pass
            demo.close()


if __name__ == "__main__":
    main()
