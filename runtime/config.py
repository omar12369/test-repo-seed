import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_ENV = os.getenv("APP_ENV", "dev")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
