# runtime/log_setup.py
import logging
from pathlib import Path


def setup_logger(name: str, level: str = "INFO", log_dir: str = "logs"):
    # ensure log_dir is a Path (not a plain string)
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # avoid duplicate handlers when uvicorn reloads
    if not logger.handlers:
        fh = logging.FileHandler(log_path / f"{name}.log")
        ch = logging.StreamHandler()

        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
