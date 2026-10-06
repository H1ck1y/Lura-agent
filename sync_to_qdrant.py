"""Copy product records from MariaDB into a local Qdrant collection."""

import os

import pymysql
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


COLLECTION = os.getenv("QDRANT_COLLECTION", "luya_products")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")


def main() -> None:
    connection = pymysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", "phase0-local-only"),
        database=os.getenv("DB_NAME", "luya_shop"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT id, name, species, water_type, bait_type, brand, price, stock, description "
                "FROM products ORDER BY id"
            )
            products = cursor.fetchall()
    finally:
        connection.close()

    if not products:
        raise RuntimeError("No products found in MariaDB; load seed_data.sql first.")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        [product["description"] for product in products],
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
        trust_env=False,
    )
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=model.get_embedding_dimension(),
                distance=models.Distance.COSINE,
            ),
            on_disk_payload=True,
        )

    points = [
        models.PointStruct(
            id=int(product["id"]),
            vector=vector.tolist(),
            payload={
                "name": product["name"],
                "species": product["species"],
                "water_type": product["water_type"],
                "bait_type": product["bait_type"],
                "brand": product["brand"],
                "price": float(product["price"]),
                "stock": product["stock"],
                "description": product["description"],
            },
        )
        for product, vector in zip(products, vectors, strict=True)
    ]
    client.upsert(collection_name=COLLECTION, points=points, wait=True)
    print(f"Synced {len(points)} products into Qdrant collection '{COLLECTION}'.")


if __name__ == "__main__":
    main()
