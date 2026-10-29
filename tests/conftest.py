from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(autouse=True)
def mock_build_rag_chain():
    """Не подключаем реальные LLM, embedder и Qdrant в тестах."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Ridge uses L2 penalty [1]."

    mock_retriever = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {
        "source": "https://scikit-learn.org/stable/linear.html"
    }
    mock_doc.page_content = "Ridge regression addresses..."
    mock_retriever.invoke.return_value = [mock_doc]

    with patch(
        "app.main.build_rag_chain",
        return_value=(mock_chain, mock_retriever),
    ) as mock:
        yield mock


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c