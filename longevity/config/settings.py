# longevity/config/settings.py
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    # Project paths
    project_root: Path = Path(__file__).parent.parent.parent
    data_dir: Path = project_root / "data"
    models_dir: Path = project_root / "models"

    # API settings
    entrez_email: Optional[str] = None

    # Model settings
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    generation_model: str = "distilgpt2"

    # RAG settings
    index_path: Path = data_dir / "index" / "faiss_index.faiss"
    top_k_retrieval: int = 5

    # Fine-tuning settings
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 2

    # GPU settings
    use_cuda: bool = True
    max_gpu_memory: str = "4GB"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
