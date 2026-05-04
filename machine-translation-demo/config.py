from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    page_title: str = "机器翻译对比与评测系统"
    model_name: str = "Helsinki-NLP/opus-mt-en-zh"
    max_new_tokens: int = 128
    chunk_chars: int = 500
    use_gpu_if_available: bool = True
    dictionary_path: str = "assets/dictionary.json"
    examples_dir: str = "assets/examples"


DEFAULT_CONFIG = AppConfig()

