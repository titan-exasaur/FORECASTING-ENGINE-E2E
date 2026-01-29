import os
import logging
from datetime import datetime


def app_logger(name: str) -> logging.Logger:
    """
    Creates and returns a logger instance with:
    - one log file per run
    - shared RUN_ID across the application
    """

    # Read RUN_ID from environment (set once by app)
    run_id = os.getenv("RUN_ID")
    if not run_id:
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )

    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"run_{run_id}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logger initialized | run_id={run_id}")

    return logger
