APP_ENV=dev
LOG_LEVEL=INFO
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
APP_ENV = os.getenv("APP_ENV", "dev")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
from dotenv import load_dotenv
from pathlib import Path
from .config import APP_ENV
from .log_setup import setup_logger

def main():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    logger = setup_logger()
    logger.info("Hello from SEED — environment=%s", APP_ENV)

if __name__ == "__main__":
    main()
