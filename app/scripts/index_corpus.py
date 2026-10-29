
"""Безопасная индексация корпуса в Qdrant.

Новая коллекция строится отдельно. Alias переключается
только после успешной индексации и sanity-проверок.
"""

import json
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
    Distance,
    VectorParams,
)

from app.config import settings


CHUNKS_PATH = Path("data/corpus_chunks.jsonl")


def load_chunks(path: Path) -> list[Document]:
    """Загрузить обработанные чанки из JSONL."""
    docs = []

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue

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
    """Построить новую коллекцию и безопасно переключить алиас."""
    client = QdrantClient(url=settings.qdrant_url)

    chunks = load_chunks(CHUNKS_PATH)

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={
            "normalize_embeddings": settings.normalize_embeddings,
        },
    )

    # Создаём отдельную коллекцию, не затрагивая текущий индекс.
    new_collection = (
        f"{settings.collection_name}_build_{uuid.uuid4().hex[:12]}"
    )

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

        # Проверяем, что коллекция содержит векторы.
        info = client.get_collection(new_collection)

        if not info.points_count or info.points_count <= 0:
            raise RuntimeError(
                "Новая коллекция не содержит векторов"
            )

        # Проверяем поиск до переключения алиаса.
        for query in [
            "How does Ridge regression work?",
            "Что такое переобучение?",
            "what do you know about?",
        ]:
            hits = vectorstore.similarity_search(query, k=3)

            if not hits:
                raise RuntimeError(
                    f"Проверочный запрос не дал результатов: {query}"
                )

            print(f"Sanity OK: {query!r}, hits={len(hits)}")

        # Проверяем, существует ли рабочий алиас.
        aliases = client.get_aliases().aliases

        alias_exists = any(
            item.alias_name == settings.collection_name
            for item in aliases
        )

        # Формируем корректные операции Qdrant.
        operations = []

        if alias_exists:
            operations.append(
                DeleteAliasOperation(
                    delete_alias=DeleteAlias(
                        alias_name=settings.collection_name
                    )
                )
            )

        operations.append(
            CreateAliasOperation(
                create_alias=CreateAlias(
                    collection_name=new_collection,
                    alias_name=settings.collection_name,
                )
            )
        )

        # Переключение алиаса выполняется одной операцией.
        client.update_collection_aliases(
            change_aliases_operations=operations
        )

        print(
            f"Alias {settings.collection_name} now points to "
            f"{new_collection}"
        )

        # Старые коллекции пока не удаляем:
        # сначала проверяем работоспособность приложения.

    except Exception:
        # Если новая коллекция не стала целью алиаса,
        # удаляем её как неудачную сборку.
        # Рабочую коллекцию не трогаем.
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