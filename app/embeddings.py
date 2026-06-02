import argparse
import hashlib
from pathlib import Path

from app.config import Settings
from app.openai_client import get_openai_client
from app.pinecone_client import get_index


def embed_text(settings: Settings, text: str) -> list[float]:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY or AZURE_OPENAI_API_KEY is required for text queries")

    client = get_openai_client(settings)
    request = {
        "model": settings.azure_openai_embedding_deployment,
        "input": text,
    }
    embedding_dimensions = getattr(settings, "azure_openai_embedding_dimensions", 1024)
    if embedding_dimensions > 0:
        request["dimensions"] = embedding_dimensions

    response = client.embeddings.create(**request)
    return response.data[0].embedding


def _chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    cleaned = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(cleaned):
        end = start + chunk_size
        chunks.append(cleaned[start:end])
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def ingest_docs(settings: Settings, docs_dir: Path, chunk_size: int, overlap: int) -> int:
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required")
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY or AZURE_OPENAI_API_KEY is required")

    index = get_index(settings)
    total = 0
    for path in sorted(docs_dir.glob("*.txt")):
        for number, chunk in enumerate(_chunks(path.read_text(encoding="utf-8"), chunk_size, overlap), start=1):
            digest = hashlib.sha256(f"{path.name}:{number}:{chunk}".encode()).hexdigest()[:32]
            index.upsert(
                vectors=[
                    {
                        "id": digest,
                        "values": embed_text(settings, chunk),
                        "metadata": {
                            "source_file": path.name,
                            "chunk_number": number,
                            "title": path.stem.replace("-", " ").replace("_", " ").title(),
                            "text": chunk,
                        },
                    }
                ],
                namespace=settings.pinecone_namespace,
            )
            total += 1
    if not total:
        raise RuntimeError(f"No .txt document content found in {docs_dir}")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed and ingest .txt documents into Pinecone.")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()
    settings = Settings()
    total = ingest_docs(settings, Path(args.docs_dir), args.chunk_size, args.overlap)
    print(f"Ingested {total} chunks into Pinecone namespace '{settings.pinecone_namespace}'.")


if __name__ == "__main__":
    main()
