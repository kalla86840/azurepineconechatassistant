import argparse
import time

from pinecone import Pinecone, ServerlessSpec

from app.config import Settings


def get_index(settings: Settings):
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required")

    client = Pinecone(api_key=settings.pinecone_api_key)
    if settings.pinecone_host:
        return client.Index(host=settings.pinecone_host.rstrip("/"))
    return client.Index(settings.pinecone_index)


def _value_from(description, key: str):
    if isinstance(description, dict):
        return description.get(key)
    return getattr(description, key)


def ensure_index(settings: Settings, cloud: str, region: str, timeout_seconds: int) -> str:
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required")

    client = Pinecone(api_key=settings.pinecone_api_key)
    if settings.pinecone_index not in client.list_indexes().names():
        client.create_index(
            name=settings.pinecone_index,
            dimension=settings.azure_openai_embedding_dimensions,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        description = client.describe_index(settings.pinecone_index)
        dimension = _value_from(description, "dimension")
        if dimension != settings.azure_openai_embedding_dimensions:
            raise RuntimeError(
                f"Pinecone index '{settings.pinecone_index}' has dimension {dimension}; "
                f"expected {settings.azure_openai_embedding_dimensions}."
            )
        status = _value_from(description, "status")
        ready = status.get("ready") if isinstance(status, dict) else getattr(status, "ready", False)
        if ready:
            return _value_from(description, "host")
        time.sleep(5)
    raise TimeoutError(f"Pinecone index '{settings.pinecone_index}' was not ready after {timeout_seconds} seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or validate the Pinecone serverless index.")
    parser.add_argument("--cloud", default="aws")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    print(ensure_index(Settings(), args.cloud, args.region, args.timeout_seconds))


if __name__ == "__main__":
    main()
