from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """Configuration settings for the Fertiscan pipeline."""

    # Document Intelligence
    document_api_endpoint: str = Field(..., alias="AZURE_API_ENDPOINT")
    document_api_key: SecretStr = Field(..., alias="AZURE_API_KEY")

    # OpenAI
    llm_api_endpoint: str = Field(..., alias="AZURE_OPENAI_ENDPOINT")
    llm_api_key: SecretStr = Field(..., alias="AZURE_OPENAI_KEY")
    llm_api_deployment: str = Field(..., alias="AZURE_OPENAI_DEPLOYMENT")

    # Embeddings
    llm_embedding_api_endpoint: Optional[str] = Field(None, alias="AZURE_EMBEDDING_ENDPOINT")
    llm_embedding_api_key: Optional[SecretStr] = Field(None, alias="AZURE_EMBEDDING_KEY")

    # Optional fields
    otel_exporter_otlp_endpoint: Optional[str] = Field(None, alias="OTEL_EXPORTER_OTLP_ENDPOINT")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
