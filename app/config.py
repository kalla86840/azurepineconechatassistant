import json
from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _secret_string_value(secret_string: str, keys: tuple[str, ...]) -> str:
    try:
        payload = json.loads(secret_string)
    except json.JSONDecodeError:
        return secret_string

    if not isinstance(payload, dict):
        return secret_string

    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return secret_string


def _read_aws_secret(secret_arn: str, keys: tuple[str, ...]) -> str:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required when PINECONE_API_KEY_SECRET_ARN is configured") from exc

    response = boto3.client("secretsmanager").get_secret_value(SecretId=secret_arn)
    secret_string = response.get("SecretString")
    if not secret_string:
        raise RuntimeError(f"Secret {secret_arn} did not include a SecretString value")

    return _secret_string_value(secret_string, keys)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "azure-pinecone-chat"
    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    pinecone_api_key_secret_arn: str = Field(default="", alias="PINECONE_API_KEY_SECRET_ARN")
    pinecone_host: str = Field(default="", alias="PINECONE_HOST")
    pinecone_index: str = Field(default="azure-pinecone-chat-1024", alias="PINECONE_INDEX")
    pinecone_namespace: str = Field(default="knowledge", alias="PINECONE_NAMESPACE")
    pinecone_memory_namespace: str = Field(default="memory", alias="PINECONE_MEMORY_NAMESPACE")
    azure_openai_endpoint: str = Field(default="", alias="AZURE_OPENAI_ENDPOINT")
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"),
    )
    azure_openai_api_key_secret_arn: str = Field(default="", alias="AZURE_OPENAI_API_KEY_SECRET_ARN")
    azure_openai_api_version: str = Field(default="2024-02-01", alias="AZURE_OPENAI_API_VERSION")
    azure_openai_embedding_deployment: str = Field(
        default="text-embedding-3-small",
        alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    )
    azure_openai_embedding_dimensions: int = Field(
        default=1024,
        alias="AZURE_OPENAI_EMBEDDING_DIMENSIONS",
    )
    openai_chat_model: str = Field(
        default="gpt-5.5",
        validation_alias=AliasChoices("OPENAI_MODEL", "AZURE_OPENAI_CHAT_DEPLOYMENT"),
    )
    rag_max_context_chars: int = Field(default=12000, alias="RAG_MAX_CONTEXT_CHARS")

    @model_validator(mode="after")
    def resolve_secret_backed_api_keys(self) -> "Settings":
        if not self.pinecone_api_key and self.pinecone_api_key_secret_arn:
            self.pinecone_api_key = _read_aws_secret(
                self.pinecone_api_key_secret_arn,
                ("PINECONE_API_KEY", "pinecone_api_key", "api_key"),
            )
        if not self.openai_api_key and self.azure_openai_api_key_secret_arn:
            self.openai_api_key = _read_aws_secret(
                self.azure_openai_api_key_secret_arn,
                ("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY", "azure_openai_api_key", "openai_api_key", "api_key"),
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
