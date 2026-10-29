
"""Безопасная индексация корпуса в Qdrant.

Новая коллекция строится отдельно. Alias sklearn_docs переключается
только после успешной индексации и sanity-проверок.
"""
import json
import time
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAlias,
    DeleteAlias,
    Distance,
    VectorParams,
)

from app.config import settings

CHUNKS_PATH = Path("data/corpus_chunks.jsonl")


def load_chunks(path: Path) -> list[Document]:
    docs = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            docs.append(
                Document(
                    page_content=row["content"],
                    metadata=row["metadata"],
                )
            )

    if not docs:
        raise RuntimeError("Корпус пуст: индексация отменена")

    print(f"Loaded {len(docs)} chunks from {path}")
    return docs


def main() -> None:
    client = QdrantClient(url=settings.qdrant_url)

    chunks = load_chunks(CHUNKS_PATH)
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={
            "normalize_embeddings": settings.normalize_embeddings
        },
    )

    # Уникальное имя: текущая рабочая коллекция не затрагивается.
    new_collection = f"{settings.collection_name}_build_{uuid.uuid4().hex[:12]}"

    client.create_collection(
        collection_name=new_collection,
        vectors_config=VectorParams(
            size=settings.embedding_dim,
            distance=Distance.COSINE,
        ),
    )

    try:
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=new_collection,
            embedding=embeddings,
        )

        vectorstore.add_documents(chunks)
        print(f"Indexed {len(chunks)} chunks into {new_collection}")

        # Проверяем новую коллекцию до переключения.
        info = client.get_collection(new_collection)
        if not info.points_count or info.points_count <= 0:
            raise RuntimeError("Новая коллекция не содержит векторов")

        for query in [
            "How does Ridge regression work?",
            "Что такое переобучение?",
            "what do you know about?",
        ]:
            hits = vectorstore.similarity_search(query, k=3)
            if not hits:
                raise RuntimeError(f"Проверочный запрос не дал результатов: {query}")
            print(f"Sanity OK: {query!r}, hits={len(hits)}")

        # Qdrant применяет операции с alias атомарно.
        aliases = client.get_aliases().aliases
        alias_exists = any(
            item.alias_name == settings.collection_name
            for item in aliases
        )

        operations = []
        if alias_exists:
            operations.append(
                DeleteAlias(alias_name=settings.collection_name)
            )

        operations.append(
            CreateAlias(
                collection_name=new_collection,
                alias_name=settings.collection_name,
            )
        )

        client.update_collection_aliases(
            change_aliases=operations
        )

        print(
            f"Alias {settings.collection_name} now points to "
            f"{new_collection}"
        )

        # Старые коллекции намеренно не удаляем здесь:
        # сначала нужно проверить новую версию API.

    except Exception:
        # Рабочий alias не менялся, поэтому старый индекс остаётся доступным.
        # Не удаляем коллекцию, если alias уже успел переключиться на неё.
        current_aliases = client.get_aliases().aliases
        current_target = next(
            (
                item.collection_name
                for item in current_aliases
                if item.alias_name == settings.collection_name
            ),
            None,
        )

        if current_target != new_collection:
            client.delete_collection(new_collection)

        raise


if __name__ == "__main__":
    main()