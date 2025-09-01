from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Core
    AWS_REGION: str
    AWS_S3_BUCKET_NAME: str
    S3_PREFIX: str

    # Models
    EMBEDDING_MODEL_ID: str
    LLM_MODEL_ID: str

    # Storage (used only if local persist is enabled)
    VAR_DIR: str = "var"
    TEXT_DIR: str = "var/text"
    TABLE_DIR: str = "var/tables"
    INDEX_PATH: str = "var/faiss.index"
    META_PATH: str = "var/index_items.json"

    # Persistence toggle (set to True on EC2 for in-memory only)
    DISABLE_LOCAL_PERSIST: bool = False

    class Config:
        env_file = "../.env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
