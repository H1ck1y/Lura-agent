"""Read-only retrieval ablation, NOT an answer-quality or production benchmark."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import time

from knowledge_tool import KnowledgeRAGTool, SharedEmbedder
from phase1_config import InventoryError, Settings
from rag_retrieval import LocalReranker, RetrievalOptions


def metrics(rows):
    positives = [r for r in rows if r["relevant"]]
    negatives = [r for r in rows if not r["relevant"]]
    def mean(values):
        return round(statistics.mean(values), 4) if values else None
    return {
        "positive_count": len(positives), "negative_count": len(negatives),
        "hit_at_1": mean([bool(r["retrieved"] and r["retrieved"][0] in r["relevant"]) for r in positives]),
        "recall_at_3": mean([len(set(r["retrieved"]) & set(r["relevant"])) / len(r["relevant"]) for r in positives]),
        "mrr_at_3": mean([next((1 / rank for rank, name in enumerate(r["retrieved"], 1) if name in r["relevant"]), 0)
                          for r in positives]),
        "negative_abstention_rate": mean([r["status"] == "empty" for r in negatives]),
        "median_elapsed_ms": round(statistics.median(r["retrieval"]["elapsed_ms"] for r in rows), 2) if rows else None,
        "positive_mean_snippets": mean([len(r["retrieved"]) for r in positives]),
        "extra_llm_calls": sum(r["retrieval"]["extra_llm_calls"] for r in rows),
        "extra_prompt_tokens": sum(r["retrieval"]["extra_prompt_tokens"] for r in rows),
        "extra_completion_tokens": sum(r["retrieval"]["extra_completion_tokens"] for r in rows),
        "usage_complete": all(r["retrieval"]["usage_complete"] for r in rows),
        "warning_count": sum(len(r["retrieval"]["warnings"]) for r in rows),
    }


def evaluate(settings, cases, variants):
    embedder = SharedEmbedder(settings.embedding_model)
    base = RetrievalOptions.from_env()
    ranker = LocalReranker(base.reranker_model)
    configurations = {
        "basic": replace(base, mode="basic", rerank=False, mqe=False, hyde=False),
        "hybrid": replace(base, mode="hybrid", rerank=False, mqe=False, hyde=False),
        "rerank": replace(base, mode="hybrid", rerank=True, mqe=False, hyde=False),
        "mqe": replace(base, mode="hybrid", rerank=True, mqe=True, hyde=False),
        "hyde": replace(base, mode="hybrid", rerank=True, mqe=False, hyde=True),
        "both": replace(base, mode="hybrid", rerank=True, mqe=True, hyde=True),
    }
    report = {"scope": "tiny curated retrieval smoke test; not a held-out benchmark or answer evaluation",
              "cost_note": "API token usage only; not an invoice or dollar estimate. Local CPU/model download cost not priced.",
              "embedding_model": settings.embedding_model, "reranker_model": base.reranker_model,
              "variants": {}}
    for name in variants:
        tool = KnowledgeRAGTool(settings, embedder=embedder, reranker=ranker, options=configurations[name])
        try:
            products_before = tool.client.count(settings.collection, exact=True).count
            # Explicit warmup uses no LLM, excluded from measured requests/costs.
            actual_options = tool.options
            tool.options = replace(actual_options, mqe=False, hyde=False)
            warm_started = time.perf_counter()
            tool.search_result("铅头钩搭配软饵挂底")
            warmup_ms = round((time.perf_counter() - warm_started) * 1000, 2)
            tool.options = actual_options
            rows = []
            for case in cases:
                result = tool.search_result(case["query"])
                rows.append({**case, "retrieved": [s["source_file"] for s in result["snippets"]],
                             "status": result["status"], "retrieval": result["retrieval"]})
            if tool.client.count(settings.collection, exact=True).count != products_before:
                raise InventoryError("商品数量变化，已停止评估。")
            report["variants"][name] = {"metrics": metrics(rows), "rows": rows, "warmup_ms": warmup_ms,
                                         "corpus_revision": tool.revision, "options": {
                                             "mode": actual_options.mode, "rerank": actual_options.rerank,
                                             "mqe": actual_options.mqe, "hyde": actual_options.hyde}}
        finally:
            tool.close()
    return report


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Phase 3.1 只读检索评估，不生成最终回答")
    parser.add_argument("--with-llm", action="store_true", help="允许收费 API 扩展；默认增加 MQE/HyDE/both 三种对照，建议先 --limit 2")
    parser.add_argument("--variants", nargs="+", choices=["basic", "hybrid", "rerank", "mqe", "hyde", "both"])
    parser.add_argument("--limit", type=int, help="每种模式使用相同的前 N 道题")
    parser.add_argument("--summary", action="store_true", help="仅输出指标，不输出逐题结果")
    args = parser.parse_args()
    variants = args.variants or (["basic", "hybrid", "rerank", "mqe", "hyde", "both"] if args.with_llm else ["basic", "hybrid", "rerank"])
    if any(v in {"mqe", "hyde", "both"} for v in variants) and not args.with_llm:
        parser.error("扩展评估需要显式 --with-llm，会产生 API 费用。")
    cases = json.loads((Path(__file__).resolve().parent / "evals" / "rag_cases.json").read_text(encoding="utf-8"))
    if args.limit is not None:
        if not 1 <= args.limit <= len(cases):
            parser.error("--limit 超出题目数量范围。")
        cases = cases[:args.limit]
    settings = Settings.from_env()
    if args.with_llm:
        settings.require_llm()
    report = evaluate(settings, cases, list(dict.fromkeys(variants)))
    if args.summary:
        for variant in report["variants"].values():
            variant.pop("rows")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except InventoryError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
