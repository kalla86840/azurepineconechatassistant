import os
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from app.config import Settings, get_settings
from app.embeddings import embed_text
from app.memory import build_memory_vector, memory_filter
from app.pinecone_client import get_index
from app.rag import generate_answer


class QueryRequest(BaseModel):
    vector: list[float] | None = Field(default=None, min_length=1)
    text: str | None = Field(default=None, min_length=1)
    question: str | None = Field(default=None, min_length=1)
    mode: Literal["search", "rag", "diagnostics", "duplicates"] = "search"
    top_k: int = Field(default=10, ge=1, le=100)
    include_metadata: bool = True
    include_values: bool = False
    filter: dict[str, Any] | None = None
    namespace: str | None = None

    @model_validator(mode="after")
    def require_text_or_vector(self) -> "QueryRequest":
        if self.vector is None and not self.text and not self.question:
            raise ValueError("Either text, question, or vector is required")
        return self

    def search_text(self) -> str:
        return self.text or self.question or ""


class DuplicateRequest(QueryRequest):
    threshold: float = Field(default=0.95, ge=0, le=1)
    exclude_ids: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("exclude_ids")
    @classmethod
    def exclude_ids_must_not_be_blank(cls, ids: list[str]) -> list[str]:
        if any(not item.strip() for item in ids):
            raise ValueError("exclude_ids cannot contain blank values")
        return ids


class FetchRequest(BaseModel):
    ids: list[str] = Field(..., min_length=1, max_length=100)
    namespace: str | None = None

    @field_validator("ids")
    @classmethod
    def ids_must_not_be_blank(cls, ids: list[str]) -> list[str]:
        if any(not item.strip() for item in ids):
            raise ValueError("ids cannot contain blank values")
        return ids


class MemoryStoreRequest(BaseModel):
    text: str = Field(..., min_length=1)
    user_id: str | None = Field(default=None, min_length=1, max_length=128)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)
    namespace: str | None = None


class MemorySearchRequest(BaseModel):
    text: str = Field(..., min_length=1)
    user_id: str | None = Field(default=None, min_length=1, max_length=128)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    top_k: int = Field(default=10, ge=1, le=100)
    include_metadata: bool = True
    include_values: bool = False
    filter: dict[str, Any] | None = None
    namespace: str | None = None


app = FastAPI(
    title="Azure Pinecone Chat",
    version="1.0.0",
    description="Azure-hosted OpenAI and Pinecone RAG chat endpoint.",
)


@app.get("/", include_in_schema=False)
def chat_page() -> FileResponse:
    return FileResponse("app/static/index.html")


def pinecone_failure_detail(
    operation: str,
    exc: Exception,
    settings: Settings,
    namespace: str,
) -> str:
    return (
        f"Pinecone {operation} failed: {exc}. "
        "Check that PINECONE_API_KEY belongs to the same Pinecone project as "
        f"PINECONE_INDEX='{settings.pinecone_index}' and "
        f"PINECONE_HOST='{str(settings.pinecone_host).rstrip('/')}'. "
        f"Request namespace was '{namespace}'."
    )


def embedding_failure_detail(exc: Exception, settings: Settings) -> str:
    return (
        f"Embedding generation failed: {exc}. "
        "Check OPENAI_API_KEY and, when using Azure OpenAI, AZURE_OPENAI_ENDPOINT, "
        f"AZURE_OPENAI_API_VERSION='{settings.azure_openai_api_version}' is supported, "
        f"AZURE_OPENAI_EMBEDDING_DEPLOYMENT='{settings.azure_openai_embedding_deployment}' is the exact deployment name, "
        f"and AZURE_OPENAI_EMBEDDING_DIMENSIONS='{settings.azure_openai_embedding_dimensions}' matches the Pinecone index dimension."
    )


def memory_namespace(request_namespace: str | None, settings: Settings) -> str:
    return request_namespace or settings.pinecone_memory_namespace


@app.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    return {
        "status": "ok",
        "app": settings.app_name,
        "build": os.environ.get("APP_BUILD_ID", "local"),
        "pinecone": {
            "index": settings.pinecone_index,
            "namespace": settings.pinecone_namespace,
            "memory_namespace": settings.pinecone_memory_namespace,
            "host": settings.pinecone_host,
            "configured": bool(settings.pinecone_api_key),
        },
    }


@app.get("/ping")
def sagemaker_ping(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    return health(settings=settings)


@app.post("/query")
def query_pinecone(
    request: QueryRequest,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    namespace = request.namespace or settings.pinecone_namespace
    vector = request.vector

    if vector is None:
        try:
            vector = embed_text(settings, request.search_text())
        except Exception as exc:
            raise HTTPException(status_code=502, detail=embedding_failure_detail(exc, settings)) from exc

    try:
        index = get_index(settings)
        result = index.query(
            vector=vector,
            top_k=request.top_k,
            namespace=namespace,
            include_metadata=request.include_metadata,
            include_values=request.include_values,
            filter=request.filter,
        )
    except Exception as exc:
        detail = pinecone_failure_detail("query", exc, settings, namespace)
        raise HTTPException(status_code=502, detail=detail) from exc

    return result.to_dict() if hasattr(result, "to_dict") else dict(result)


@app.post("/rag")
def rag_pinecone(
    request: QueryRequest,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    question = request.question or request.text
    if not question:
        raise HTTPException(status_code=422, detail="question or text is required for RAG")

    search_request = request.model_copy(
        update={
            "text": question,
            "include_metadata": True,
            "include_values": False,
        }
    )
    search_result = query_pinecone(request=search_request, settings=settings)
    matches = search_result.get("matches", [])

    try:
        answer = generate_answer(settings=settings, question=question, matches=matches)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"RAG answer generation failed: {exc}") from exc

    return {
        "question": question,
        "answer": answer["answer"],
        "sources": answer["sources"],
        "matches": matches,
    }


@app.post("/memory")
def store_memory(
    request: MemoryStoreRequest,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    namespace = memory_namespace(request.namespace, settings)

    try:
        embedding = embed_text(settings, request.text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=embedding_failure_detail(exc, settings)) from exc

    try:
        vector = build_memory_vector(
            settings=settings,
            text=request.text,
            embedding=embedding,
            user_id=request.user_id,
            session_id=request.session_id,
            metadata=request.metadata,
        )
        index = get_index(settings)
        index.upsert(vectors=[vector], namespace=namespace)
    except Exception as exc:
        detail = pinecone_failure_detail("memory upsert", exc, settings, namespace)
        raise HTTPException(status_code=502, detail=detail) from exc

    return {
        "id": vector["id"],
        "namespace": namespace,
        "metadata": vector["metadata"],
    }


@app.post("/memory/search")
def search_memory(
    request: MemorySearchRequest,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    namespace = memory_namespace(request.namespace, settings)

    try:
        vector = embed_text(settings, request.text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=embedding_failure_detail(exc, settings)) from exc

    try:
        index = get_index(settings)
        result = index.query(
            vector=vector,
            top_k=request.top_k,
            namespace=namespace,
            include_metadata=request.include_metadata,
            include_values=request.include_values,
            filter=memory_filter(
                user_id=request.user_id,
                session_id=request.session_id,
                extra_filter=request.filter,
            ),
        )
    except Exception as exc:
        detail = pinecone_failure_detail("memory query", exc, settings, namespace)
        raise HTTPException(status_code=502, detail=detail) from exc

    body = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    return {
        "query": request.text,
        "namespace": namespace,
        "matches": body.get("matches", []),
    }


@app.post("/duplicates")
def detect_duplicates(
    request: DuplicateRequest,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    search_request = QueryRequest(
        vector=request.vector,
        text=request.text,
        question=request.question,
        top_k=request.top_k,
        include_metadata=True,
        include_values=request.include_values,
        filter=request.filter,
        namespace=request.namespace,
    )
    search_result = query_pinecone(request=search_request, settings=settings)
    excluded = {item.strip() for item in request.exclude_ids}
    matches = [
        match
        for match in search_result.get("matches", [])
        if match.get("score", 0) >= request.threshold and match.get("id") not in excluded
    ]

    return {
        "is_duplicate": bool(matches),
        "threshold": request.threshold,
        "duplicates": matches,
        "matches": search_result.get("matches", []),
    }


@app.post("/score")
def score(
    request: DuplicateRequest,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    if request.mode == "diagnostics":
        return health(settings=settings)
    if request.mode == "rag":
        return rag_pinecone(request=request, settings=settings)
    if request.mode == "duplicates":
        return detect_duplicates(request=request, settings=settings)
    return query_pinecone(request=request, settings=settings)


@app.post("/invocations")
def sagemaker_invocations(
    request: DuplicateRequest,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    return score(request=request, settings=settings)


@app.post("/fetch")
def fetch_records(
    request: FetchRequest,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    namespace = request.namespace or settings.pinecone_namespace

    try:
        index = get_index(settings)
        result = index.fetch(ids=request.ids, namespace=namespace)
    except Exception as exc:
        detail = pinecone_failure_detail("fetch", exc, settings, namespace)
        raise HTTPException(status_code=502, detail=detail) from exc

    return result.to_dict() if hasattr(result, "to_dict") else dict(result)
