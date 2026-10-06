"""Acceptance query: hard filters plus semantic ranking over descriptions."""

import os
import sys

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


COLLECTION = os.getenv("QDRANT_COLLECTION", "luya_products")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    model = SentenceTransformer(MODEL_NAME)
    query_vector = model.encode(
        "新手友好，容易上手，不易挂底", normalize_embeddings=True
    ).tolist()
    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
        trust_env=False,
    )
    result = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="species", match=models.MatchValue(value="鲈鱼")
                ),
                models.FieldCondition(key="price", range=models.Range(lte=100)),
                models.FieldCondition(key="stock", range=models.Range(gt=0)),
            ]
        ),
        limit=10,
        with_payload=True,
    )
    points = result.points
    if not points:
        raise AssertionError("Acceptance query returned no in-stock bass products under 100.")

    print("Query: 鲈鱼 + 预算 100 + 有库存 + 新手友好/不易挂底（语义排序）")
    for rank, point in enumerate(points, start=1):
        payload = point.payload or {}
        print(
            f"{rank}. {payload['name']} | {payload['species']} | "
            f"CNY {payload['price']:.2f} | stock={payload['stock']} | score={point.score:.4f}"
        )

    assert all(
        point.payload["species"] == "鲈鱼"
        and point.payload["price"] <= 100
        and point.payload["stock"] > 0
        for point in points
    ), "A result did not satisfy all structured filters."


if __name__ == "__main__":
    main()
