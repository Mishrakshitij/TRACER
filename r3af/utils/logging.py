import logging
import os
from typing import Optional, Dict, Any


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_config: Optional[Dict[str, Any]] = None,
) -> logging.Logger:
    logger = logging.getLogger("r3af")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if wandb_project:
        try:
            import wandb
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=wandb_config or {},
                reinit=True,
            )
            logger.info(f"Wandb initialized: project={wandb_project}")
        except ImportError:
            logger.warning("wandb not installed, skipping wandb init")

    return logger
