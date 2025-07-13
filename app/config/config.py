"""Module for application configuration"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(".") / ".env"
load_dotenv(env_path)


@dataclass
class Settings:
    """Class to hold application settings"""

    project_name: str = os.getenv("PROJECT_NAME") or "meta_learning_mvp"
    project_version: str = os.getenv("PROJECT_VERSION") or "0.0.0"
    log_level: str = os.getenv("LOG_LEVEL") or "INFO"


settings = Settings()
