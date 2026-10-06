"""Project-local configuration shared by the Phase 1 scripts."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


class InventoryError(RuntimeError):
    """An actionable inventory or model-service failure."""


@dataclass(frozen=True)
class Settings:
    db_host: str = "127.0.0.1"
    db_port: int = 3306
    db_user: str = "root"
    db_password: str = field(default="phase0-local-only", repr=False)
    db_name: str = "luya_shop"
    qdrant_url: str = "http://127.0.0.1:6333"
    collection: str = "luya_products"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key: str = field(default="", repr=False)
    llm_timeout: float = 60
    llm_json_mode: bool = True
    llm_extra_body: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_env(cls):
        load_dotenv(Path(__file__).with_name(".env"), override=False)
        try:
            extra = json.loads(os.getenv("LLM_EXTRA_BODY", "{}"))
            if not isinstance(extra, dict):
                raise ValueError("LLM_EXTRA_BODY must be a JSON object")
            if set(extra) & {"model", "messages", "response_format", "stream"}:
                raise ValueError("LLM_EXTRA_BODY cannot override core request fields")
            timeout = float(os.getenv("LLM_TIMEOUT", "60"))
            if not 0 < timeout <= 600:
                raise ValueError("LLM_TIMEOUT must be between 0 and 600 seconds")
            json_mode = os.getenv("LLM_JSON_MODE", "true").lower()
            if json_mode not in {"true", "false"}:
                raise ValueError("LLM_JSON_MODE must be true or false")
            return cls(
                db_host=os.getenv("DB_HOST", "127.0.0.1"),
                db_port=int(os.getenv("DB_PORT", "3306")),
                db_user=os.getenv("DB_USER", "root"),
                db_password=os.getenv("DB_PASSWORD", "phase0-local-only"),
                db_name=os.getenv("DB_NAME", "luya_shop"),
                qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
                collection=os.getenv("QDRANT_COLLECTION", "luya_products"),
                embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
                llm_base_url=os.getenv("LLM_BASE_URL", "").strip().rstrip("/"),
                llm_model=os.getenv("LLM_MODEL", "").strip(),
                llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
                llm_timeout=timeout,
                llm_json_mode=json_mode == "true",
                llm_extra_body=extra,
            )
        except (ValueError, TypeError) as exc:
            raise InventoryError("配置无效，请检查 .env 的端口、超时和 JSON 参数。") from exc

    def require_llm(self):
        missing = [name for name, value in (
            ("LLM_BASE_URL", self.llm_base_url),
            ("LLM_MODEL", self.llm_model),
            ("LLM_API_KEY", self.llm_api_key),
        ) if not value]
        if missing:
            raise InventoryError("请在本地 .env 中配置：" + ", ".join(missing))
