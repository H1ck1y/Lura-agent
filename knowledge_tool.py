"""Read-only Hello-Agents RAGTool integration with local BGE/Qdrant adapters."""

import hashlib
import json
import math
from pathlib import Path
import time
import uuid

from hello_agents.memory.rag.pipeline import load_and_chunk_texts
from hello_agents.memory.storage.qdrant_store import QdrantVectorStore
from hello_agents.tools import RAGTool
from hello_agents.tools.base import ToolParameter
from qdrant_client import QdrantClient, models

from phase1_config import InventoryError
from agent_llm import AgentChatLLM
from rag_retrieval import ChineseBM25, LocalReranker, QueryExpansion, RetrievalOptions, reciprocal_rank_fusion


ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = ROOT / "knowledge_base"
KNOWLEDGE_COLLECTION = "luya_knowledge_phase3"
NAMESPACE = "luya_phase3"


class SharedEmbedder:
    """Lazy embedding instance shared with the inventory tool within one process."""
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def encode(self, text, **kwargs):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise InventoryError("无法加载本地 embedding 模型，请检查缓存和网络。") from exc
        return self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)


def load_sources(directory=KNOWLEDGE_DIR):
    directory = Path(directory).resolve()
    try:
        sources = json.loads((directory / "sources.json").read_text(encoding="utf-8"))
        if not isinstance(sources, dict) or not sources:
            raise ValueError("Empty manifest")
        for name, source in sources.items():
            path = directory / name
            if Path(name).name != name or path.resolve().parent != directory or path.suffix != ".md" or not path.is_file():
                raise ValueError("Unsafe/missing source")
            if not isinstance(source, dict) or not isinstance(source.get("title"), str):
                raise ValueError("Invalid source metadata")
            url = source.get("url")
            if url is not None and (not isinstance(url, str) or not url.startswith("https://")):
                raise ValueError("Source URL must be HTTPS or null")
        return sources
    except (OSError, ValueError, TypeError) as exc:
        raise InventoryError("知识库清单无效，请检查 knowledge_base/sources.json 和对应 Markdown 文件。") from exc


def corpus_revision(directory, sources, model_name):
    digest = hashlib.sha256(model_name.encode())
    for name in sorted(sources):
        digest.update(name.encode())
        digest.update(json.dumps(sources[name], ensure_ascii=False, sort_keys=True).encode())
        digest.update((Path(directory) / name).read_bytes())
    return digest.hexdigest()


class LocalKnowledgeStore(QdrantVectorStore):
    """Reuse framework storage indexing; use fail-closed reads and no proxy for localhost."""
    def __init__(self, client, collection_name=KNOWLEDGE_COLLECTION):
        self.client = client
        self.collection_name = collection_name
        self.vector_size = None

    def ensure_collection(self, vector_size):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
        config = self.client.get_collection(self.collection_name).config.params.vectors
        if not isinstance(config, models.VectorParams) or config.size != vector_size or config.distance != models.Distance.COSINE:
            raise InventoryError("知识集合向量配置不匹配，请使用独立集合重新索引，不要覆盖商品集合。")
        self.vector_size = vector_size

    def search_similar(self, query_vector, limit=3, score_threshold=0.5, where=None):
        try:
            result = self.client.query_points(
                collection_name=self.collection_name, query=query_vector,
                query_filter=models.Filter(must=[
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                    for key, value in (where or {}).items()
                ]), limit=limit, score_threshold=score_threshold, with_payload=True,
            )
            return [{"id": str(p.id), "score": p.score, "metadata": p.payload or {}} for p in result.points]
        except Exception as exc:
            raise InventoryError("知识库查询失败；请检查 Qdrant 并运行 index_knowledge.py。") from exc


class KnowledgeRAGTool(RAGTool):
    """The agent sees only query: no add/clear/file/network operations."""
    def __init__(self, settings, *, client=None, embedder=None, directory=KNOWLEDGE_DIR, min_score=0.5,
                 options=None, reranker=None, expansion_llm=None):
        if settings.collection == KNOWLEDGE_COLLECTION:
            raise InventoryError("知识集合必须与商品集合不同，请修正 QDRANT_COLLECTION。")
        self.settings = settings
        self.client = client if client is not None else QdrantClient(
            url=settings.qdrant_url, trust_env=False, timeout=15, check_compatibility=False
        )
        self.embedder = embedder or SharedEmbedder(settings.embedding_model)
        self.directory = Path(directory).resolve()
        self.sources = load_sources(self.directory)
        self.revision = corpus_revision(self.directory, self.sources, settings.embedding_model)
        self.min_score = min_score
        self.options = options or RetrievalOptions.from_env()
        self.reranker = reranker or LocalReranker(self.options.reranker_model)
        self.expansion_llm = expansion_llm or AgentChatLLM(settings)
        self.expander = QueryExpansion(self.expansion_llm)
        self._lexical_cache = None
        self.last_diagnostics = {}
        super().__init__(knowledge_base_path=str(self.directory), collection_name=KNOWLEDGE_COLLECTION,
                         rag_namespace=NAMESPACE, expandable=False)
        self.description = (
            "只读路亚知识库检索：回答拟饵区别、选色、铅头钩挂底、匀收/轻抽等钓法知识。"
            "不查询本店商品、价格或库存；选购推荐必须调用 inventory。仅接受 query 字符串。"
            "混合问题请提取知识子问题，保留与技巧相关情境，但不含商品预算、购买或库存要求。"
        )

    def _init_components(self):
        # Override only infrastructure wiring: no default cloud embedder/LLM,
        # no startup mutation of Qdrant; optional expansion is explicit and bounded.
        self.store = LocalKnowledgeStore(self.client, self.collection_name)
        self._pipelines[self.rag_namespace] = {"store": self.store, "search": self.retrieve}
        self.initialized = True

    def get_parameters(self):
        return [ToolParameter(name="query", type="string", description="完整的路亚知识问题", required=True)]

    def _where(self):
        return {"rag_namespace": NAMESPACE, "corpus_revision": self.revision,
                "embedding_model": self.settings.embedding_model}

    def _lexical_index(self, where, count):
        key = (self.revision, count)
        if self._lexical_cache is not None and self._lexical_cache[0] == key:
            return self._lexical_cache[1]
        if count > 5000:
            raise InventoryError("本地 BM25 演示索引最多支持 5000 分块；请迁移服务端稀疏索引。")
        query_filter = models.Filter(must=[
            models.FieldCondition(key=k, match=models.MatchValue(value=v)) for k, v in where.items()
        ])
        chunks, offset = [], None
        while True:
            points, offset = self.client.scroll(self.collection_name, scroll_filter=query_filter,
                                                offset=offset, limit=256, with_payload=True, with_vectors=False)
            for point in points:
                meta = point.payload or {}
                if meta.get("source_file") in self.sources and isinstance(meta.get("content"), str) and meta["content"].strip():
                    chunks.append({"id": str(point.id), "metadata": meta})
            if offset is None:
                break
        if not chunks:
            raise InventoryError("当前知识索引没有有效分块，请重新运行 index_knowledge.py。")
        index = ChineseBM25(chunks)
        self._lexical_cache = (key, index)
        return index

    def retrieve(self, query, top_k=3, score_threshold=None):
        if not isinstance(query, str) or not query.strip() or len(query) > 4000:
            raise InventoryError("知识问题不能为空，且不能超过 4000 字符。")
        if type(top_k) is not int or not 1 <= top_k <= self.options.candidate_k:
            raise InventoryError("知识 top_k 必须为 1–candidate_k 的整数。")
        started = time.perf_counter()
        threshold = self.min_score if score_threshold is None else score_threshold
        diagnostics = self.last_diagnostics = {
            "mode": self.options.mode, "rerank_enabled": self.options.rerank and self.options.mode != "basic",
            "rerank_applied": False, "mqe": self.options.mqe, "hyde": self.options.hyde,
            "extra_llm_calls": 0, "extra_prompt_tokens": 0, "extra_completion_tokens": 0,
            "usage_complete": True, "warnings": [], "candidate_count": 0,
        }
        try:
            if not self.client.collection_exists(self.collection_name):
                raise InventoryError("知识库尚未索引，请先运行 index_knowledge.py。")
            where = self._where()
            current = self.client.count(self.collection_name, exact=True, count_filter=models.Filter(must=[
                models.FieldCondition(key=key, match=models.MatchValue(value=value)) for key, value in where.items()
            ])).count
            if current == 0:
                raise InventoryError("知识文档或模型已变更，请重新运行 index_knowledge.py 更新知识索引。")
            vector = self.embedder.encode(query).tolist()
            if self.options.mode == "basic":
                hits = self.store.search_similar(vector, limit=top_k, score_threshold=threshold, where=where)
                diagnostics["candidate_count"] = len(hits)
                return [{**hit, "score_type": "cosine", "scores": {"dense": hit["score"]}} for hit in hits]
            limit = self.options.candidate_k
            dense = self.store.search_similar(vector, limit=limit, score_threshold=threshold, where=where)
            lexical = self._lexical_index(where, current)
            sparse = lexical.search(query, limit)
            rankings = [("dense", dense), ("bm25", sparse)]
            # Conservative query-level gate: do not expand a completely
            # unsupported question. Generated text itself is never evidence.
            supported = {hit["id"] for hit in dense + sparse}
            if not supported:
                diagnostics["abstained_before_expansion"] = True
                return []
            for kind in ("mqe", "hyde"):
                if not getattr(self.options, kind):
                    continue
                call_start = len(getattr(self.expansion_llm, "calls", []))
                diagnostics["extra_llm_calls"] += 1
                try:
                    expansions = self.expander.generate(query, kind)
                except Exception:
                    # Never expose provider bodies/keys; original search still works.
                    diagnostics["warnings"].append(kind.upper() + " 扩展失败，保留原问题检索；未重试。")
                    expansions = []
                records = getattr(self.expansion_llm, "calls", [])[call_start:]
                diagnostics["usage_complete"] &= bool(records) and all(
                    r.get("prompt_tokens") is not None and r.get("completion_tokens") is not None for r in records)
                for record in records:
                    diagnostics["extra_prompt_tokens"] += record.get("prompt_tokens") or 0
                    diagnostics["extra_completion_tokens"] += record.get("completion_tokens") or 0
                diagnostics[kind + "_generated_count"] = len(expansions)
                for number, text in enumerate(expansions, 1):
                    expansion_vector = self.embedder.encode(text).tolist()
                    rankings.append((f"{kind}_dense_{number}", self.store.search_similar(
                        expansion_vector, limit=limit, score_threshold=threshold, where=where)))
                    if kind == "mqe":
                        rankings.append((f"mqe_bm25_{number}", lexical.search(text, limit)))
            fused = reciprocal_rank_fusion(rankings)
            # With reranking, expansions can recall NEW indexed chunks, but the
            # original question must pass their relevance gate. Without it, only
            # original-supported candidates are eligible (conservative ablation).
            hits = (fused if self.options.rerank else [h for h in fused if h["id"] in supported])[:limit]
            diagnostics["candidate_count"] = len(hits)
            if self.options.rerank and hits:
                rank_started = time.perf_counter()
                scores = self.reranker.score(query, hits)
                if len(scores) != len(hits) or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
                    raise InventoryError("Reranker 返回了无效相关性分数。")
                for hit, score in zip(hits, scores):
                    hit["scores"]["rrf"] = hit["score"]
                    hit["score"] = float(score)
                    hit["score_type"] = "reranker_sigmoid"
                hits = [h for h in hits if h["score"] >= self.options.min_rerank_score]
                hits.sort(key=lambda h: (-h["score"], -h["scores"]["rrf"], h["id"]))
                diagnostics["rerank_applied"] = True
                diagnostics["rerank_ms"] = round((time.perf_counter() - rank_started) * 1000, 2)
            else:
                for hit in hits:
                    hit["score_type"] = "rrf"
            return hits[:top_k]
        except InventoryError:
            raise
        except Exception as exc:
            raise InventoryError("知识库服务不可用，请检查 Qdrant 和 embedding 配置。") from exc
        finally:
            diagnostics["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)

    def search_result(self, query):
        hits = self._get_pipeline(NAMESPACE)["search"](query)
        snippets = []
        for hit in hits:
            meta = hit["metadata"]
            name = meta.get("source_file")
            if name not in self.sources:
                continue
            snippets.append({
                "citation": f"K{len(snippets) + 1}", "source_file": name,
                "title": self.sources[name]["title"], "url": self.sources[name].get("url"),
                "heading": meta.get("heading_path"), "score": hit["score"],
                "score_type": hit.get("score_type", "cosine"), "scores": hit.get("scores", {}),
                "channels": hit.get("channels", ["dense"]),
                "content": meta.get("content", "")[:1500],
            })
        return {
            "status": "ok" if snippets else "empty", "query": query, "snippets": snippets,
            "retrieval": dict(self.last_diagnostics),
            "reply": "以下为知识库检索片段，不代表商品库存。" if snippets else
                     "当前知识库没有足够相关资料，暂不能依据资料回答；请补充知识文档或换一个更具体的问题。",
        }

    def run(self, parameters):
        if not isinstance(parameters, dict) or set(parameters) != {"query"}:
            raise InventoryError("知识工具只接受 query；不允许写入、清空或读取任意文件。")
        return json.dumps(self.search_result(parameters["query"]), ensure_ascii=False)

    def index_documents(self):
        paths = [str(self.directory / name) for name in sorted(self.sources)]
        chunks = load_and_chunk_texts(paths, chunk_size=500, chunk_overlap=0, namespace=NAMESPACE)
        if not chunks:
            raise InventoryError("知识文档没有产生有效分块。")
        vectors = self.embedder.encode([c["content"] for c in chunks]).tolist()
        if len(vectors) != len(chunks):
            raise InventoryError("知识向量数量与分块不一致。")
        if not vectors or any(not vector or len(vector) != len(vectors[0]) or
                              not all(math.isfinite(value) for value in vector) or
                              not any(value != 0 for value in vector) for vector in vectors):
            raise InventoryError("知识 embedding 返回了无效向量，已停止索引。")
        metadata = []
        ids = []
        for chunk in chunks:
            name = Path(chunk["metadata"]["source_path"]).name
            metadata.append({
                **chunk["metadata"], "content": chunk["content"], "source_file": name,
                "rag_namespace": NAMESPACE, "corpus_revision": self.revision,
                "embedding_model": self.settings.embedding_model, "memory_type": "rag_chunk",
                "is_rag_data": True, "data_source": "rag_pipeline",
            })
            ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, self.revision + chunk["id"])))
        try:
            self.store.ensure_collection(len(vectors[0]))
            if not self.store.add_vectors(vectors, metadata, ids):
                raise InventoryError("知识库写入失败，请检查 Qdrant。")
        except InventoryError:
            raise
        except Exception as exc:
            raise InventoryError("知识索引失败，请检查 Qdrant 服务。") from exc
        self._lexical_cache = None
        return {"documents": len(paths), "chunks": len(chunks), "collection": self.collection_name,
                "corpus_revision": self.revision}

    def close(self):
        self.client.close()
