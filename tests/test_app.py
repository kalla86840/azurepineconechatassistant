from fastapi.testclient import TestClient

from app.main import app


def test_health_uses_pinecone_defaults(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    from app.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["pinecone"]["index"] == settings.pinecone_index
    assert body["pinecone"]["namespace"] == settings.pinecone_namespace
    assert body["pinecone"]["host"] == str(settings.pinecone_host)
    assert body["pinecone"]["configured"] is False


def test_fetch_rejects_blank_ids():
    client = TestClient(app)

    response = client.post("/fetch", json={"ids": ["valid-id", " "]})

    assert response.status_code == 422


def test_openai_model_alias_sets_chat_deployment(monkeypatch):
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.delenv("AZURE_OPENAI_CHAT_DEPLOYMENT", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-nano")

    settings = get_settings()

    assert settings.openai_chat_model == "gpt-5.4-nano"


def test_root_serves_chat_page():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Knowledge chat" in response.text


def test_score_requires_text_or_vector():
    client = TestClient(app)

    response = client.post("/score", json={"top_k": 3})

    assert response.status_code == 422


def test_score_diagnostics_mode_returns_health_without_external_services(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    from app.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    client = TestClient(app)

    response = client.post(
        "/score",
        json={
            "mode": "diagnostics",
            "question": "config",
            "top_k": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["pinecone"]["index"] == settings.pinecone_index
    assert body["pinecone"]["namespace"] == settings.pinecone_namespace
    assert body["pinecone"]["configured"] is False


def test_sagemaker_ping_returns_health(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    from app.config import get_settings

    get_settings.cache_clear()
    client = TestClient(app)

    response = client.get("/ping")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_sagemaker_invocations_uses_score_contract(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    from app.config import get_settings

    get_settings.cache_clear()
    client = TestClient(app)

    response = client.post(
        "/invocations",
        json={
            "mode": "diagnostics",
            "question": "config",
            "top_k": 1,
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_pinecone_key_can_be_loaded_from_aws_secret(monkeypatch):
    from app import config
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.setenv("PINECONE_API_KEY_SECRET_ARN", "arn:aws:secretsmanager:us-east-1:123456789012:secret:pinecone")
    monkeypatch.setattr(config, "_read_aws_secret", lambda secret_arn, keys: "secret-from-manager")

    settings = get_settings()

    assert settings.pinecone_api_key == "secret-from-manager"


def test_score_accepts_text_payload(monkeypatch):
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeResult:
        def to_dict(self):
            return {"matches": []}

    class FakeIndex:
        def query(self, **kwargs):
            assert kwargs["vector"] == [0.1, 0.2, 0.3]
            return FakeResult()

    monkeypatch.setattr("app.main.embed_text", lambda settings, text: [0.1, 0.2, 0.3])
    monkeypatch.setattr("app.main.get_index", lambda settings: FakeIndex())

    client = TestClient(app)
    response = client.post("/score", json={"text": "Edgio earnings call", "top_k": 3})

    assert response.status_code == 200
    assert response.json() == {"matches": []}


def test_score_rag_mode_returns_answer(monkeypatch):
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeResult:
        def to_dict(self):
            return {
                "matches": [
                    {
                        "id": "record-1",
                        "score": 0.99,
                        "metadata": {
                            "title": "Edgio Q3 2023 Earnings Call Transcript",
                            "text": "Edgio discussed its Q3 2023 earnings call results.",
                        },
                    }
                ]
            }

    class FakeIndex:
        def query(self, **kwargs):
            assert kwargs["vector"] == [0.1, 0.2, 0.3]
            assert kwargs["include_metadata"] is True
            return FakeResult()

    monkeypatch.setattr("app.main.embed_text", lambda settings, text: [0.1, 0.2, 0.3])
    monkeypatch.setattr("app.main.get_index", lambda settings: FakeIndex())
    monkeypatch.setattr(
        "app.main.generate_answer",
        lambda settings, question, matches: {
            "answer": "Edgio discussed its Q3 2023 earnings call results. [Source 1]",
            "sources": [{"number": 1, "title": "Edgio Q3 2023 Earnings Call Transcript"}],
        },
    )

    client = TestClient(app)
    response = client.post(
        "/score",
        json={
            "mode": "rag",
            "question": "What did Edgio discuss in Q3 2023?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Edgio discussed its Q3 2023 earnings call results. [Source 1]"
    assert body["sources"][0]["number"] == 1
    assert body["matches"][0]["id"] == "record-1"


def test_duplicates_filters_matches_by_threshold_and_excluded_ids(monkeypatch):
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeResult:
        def to_dict(self):
            return {
                "matches": [
                    {"id": "dupe-1", "score": 0.98, "metadata": {"text": "duplicate"}},
                    {"id": "same-record", "score": 0.99, "metadata": {"text": "self"}},
                    {"id": "below-threshold", "score": 0.9, "metadata": {"text": "near"}},
                ]
            }

    class FakeIndex:
        def query(self, **kwargs):
            assert kwargs["vector"] == [0.1, 0.2, 0.3]
            assert kwargs["include_metadata"] is True
            return FakeResult()

    monkeypatch.setattr("app.main.embed_text", lambda settings, text: [0.1, 0.2, 0.3])
    monkeypatch.setattr("app.main.get_index", lambda settings: FakeIndex())

    client = TestClient(app)
    response = client.post(
        "/duplicates",
        json={
            "text": "possible duplicate",
            "threshold": 0.95,
            "exclude_ids": ["same-record"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["is_duplicate"] is True
    assert body["duplicates"] == [{"id": "dupe-1", "score": 0.98, "metadata": {"text": "duplicate"}}]
    assert len(body["matches"]) == 3


def test_score_duplicates_mode_accepts_threshold(monkeypatch):
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeResult:
        def to_dict(self):
            return {"matches": [{"id": "near-match", "score": 0.91}]}

    class FakeIndex:
        def query(self, **kwargs):
            return FakeResult()

    monkeypatch.setattr("app.main.embed_text", lambda settings, text: [0.1, 0.2, 0.3])
    monkeypatch.setattr("app.main.get_index", lambda settings: FakeIndex())

    client = TestClient(app)
    response = client.post(
        "/score",
        json={
            "mode": "duplicates",
            "text": "possible duplicate",
            "threshold": 0.9,
        },
    )

    assert response.status_code == 200
    assert response.json()["is_duplicate"] is True


def test_store_memory_upserts_to_memory_namespace(monkeypatch):
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured = {}

    class FakeIndex:
        def upsert(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("app.main.embed_text", lambda settings, text: [0.1, 0.2, 0.3])
    monkeypatch.setattr("app.main.get_index", lambda settings: FakeIndex())

    client = TestClient(app)
    response = client.post(
        "/memory",
        json={
            "text": "The user prefers concise status updates.",
            "user_id": "user-1",
            "session_id": "session-1",
            "metadata": {"topic": "preferences"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["namespace"] == "memory"
    assert body["metadata"]["text"] == "The user prefers concise status updates."
    assert body["metadata"]["user_id"] == "user-1"
    assert captured["namespace"] == "memory"
    assert captured["vectors"][0]["values"] == [0.1, 0.2, 0.3]
    assert captured["vectors"][0]["metadata"]["topic"] == "preferences"


def test_search_memory_scopes_by_user_and_session(monkeypatch):
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured = {}

    class FakeResult:
        def to_dict(self):
            return {"matches": [{"id": "mem-1", "score": 0.98}]}

    class FakeIndex:
        def query(self, **kwargs):
            captured.update(kwargs)
            return FakeResult()

    monkeypatch.setattr("app.main.embed_text", lambda settings, text: [0.1, 0.2, 0.3])
    monkeypatch.setattr("app.main.get_index", lambda settings: FakeIndex())

    client = TestClient(app)
    response = client.post(
        "/memory/search",
        json={
            "text": "status update preference",
            "user_id": "user-1",
            "session_id": "session-1",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    assert response.json()["matches"] == [{"id": "mem-1", "score": 0.98}]
    assert captured["namespace"] == "memory"
    assert captured["top_k"] == 3
    assert captured["filter"] == {
        "$and": [
            {"user_id": {"$eq": "user-1"}},
            {"session_id": {"$eq": "session-1"}},
        ]
    }
