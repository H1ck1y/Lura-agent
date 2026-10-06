"""Small duck-typed LLM adapter for Hello-Agents' text tool protocol."""

import time

import httpx

from phase1_config import InventoryError


class AgentChatLLM:
    def __init__(self, settings, transport=None):
        self.settings = settings
        self.transport = transport
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.settings.require_llm()
        body = {
            "model": self.settings.llm_model, "messages": messages, "stream": False,
            **self.settings.llm_extra_body,
        }
        if kwargs.get("json_mode") and self.settings.llm_json_mode:
            body["response_format"] = {"type": "json_object"}
        record = {"ok": False, "elapsed_ms": 0, "prompt_tokens": None, "completion_tokens": None}
        self.calls.append(record)
        started = time.perf_counter()
        # Agent output is text, not the slot extractor's JSON. No default temperature
        # or max_tokens from the legacy framework; preserve configured reasoning.
        try:
            with httpx.Client(timeout=self.settings.llm_timeout, transport=self.transport,
                              trust_env=self.transport is None) as client:
                response = client.post(
                    self.settings.llm_base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": "Bearer " + self.settings.llm_api_key}, json=body,
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise InventoryError(f"Agent LLM API 返回 HTTP {exc.response.status_code}；请检查模型、地址及 Key。") from exc
        except httpx.RequestError as exc:
            raise InventoryError("Agent LLM 请求失败或超时，请检查网络和配置。") from exc
        finally:
            record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        try:
            data = response.json()
            usage = data.get("usage") or {}
            for name in ("prompt_tokens", "completion_tokens"):
                value = usage.get(name) if isinstance(usage, dict) else None
                if type(value) is int and value >= 0:
                    record[name] = value
            choice = data["choices"][0]
            if choice.get("finish_reason") in {"length", "content_filter"}:
                raise InventoryError("Agent 模型输出不完整或被拦截。")
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip() or len(content) > 24000:
                raise ValueError("Invalid agent content")
            record["ok"] = True
            return content
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            raise InventoryError("Agent 模型响应格式无效，已停止本次查询。") from exc
