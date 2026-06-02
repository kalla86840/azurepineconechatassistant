import hashlib
from datetime import datetime, timezone
from typing import Any

from app.config import Settings


def memory_id(text: str, user_id: str | None, session_id: str | None, created_at: str) -> str:
    key = f"{user_id or ''}:{session_id or ''}:{created_at}:{text}"
    return f"mem-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:32]}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def memory_filter(user_id: str | None, session_id: str | None, extra_filter: dict[str, Any] | None) -> dict[str, Any] | None:
    filters: list[dict[str, Any]] = []
    if extra_filter:
        filters.append(extra_filter)
    if user_id:
        filters.append({"user_id": {"$eq": user_id}})
    if session_id:
        filters.append({"session_id": {"$eq": session_id}})

    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def build_memory_vector(
    settings: Settings,
    text: str,
    embedding: list[float],
    user_id: str | None,
    session_id: str | None,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    created_at = now_iso()
    clean_metadata = dict(metadata or {})
    clean_metadata.update(
        {
            "text": text,
            "created_at": created_at,
            "kind": "memory",
        }
    )
    if user_id:
        clean_metadata["user_id"] = user_id
    if session_id:
        clean_metadata["session_id"] = session_id

    return {
        "id": memory_id(text=text, user_id=user_id, session_id=session_id, created_at=created_at),
        "values": embedding,
        "metadata": clean_metadata,
    }
