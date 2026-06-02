from openai import AzureOpenAI, OpenAI

from app.config import Settings


def uses_azure_openai_v1(settings: Settings) -> bool:
    return settings.azure_openai_api_version.strip().lower() in {"v1", "preview"}


def azure_openai_v1_base_url(endpoint: str) -> str:
    return f"{endpoint.rstrip('/')}/openai/v1/"


def get_openai_client(settings: Settings):
    if not settings.azure_openai_endpoint:
        return OpenAI(api_key=settings.openai_api_key)

    if uses_azure_openai_v1(settings):
        return OpenAI(
            api_key=settings.openai_api_key,
            base_url=azure_openai_v1_base_url(settings.azure_openai_endpoint),
        )

    return AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.openai_api_key,
        api_version=settings.azure_openai_api_version,
    )
