import os
from loguru import logger
from torch.utils.tensorboard import SummaryWriter


def create_logger(log_dir, seed):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{seed}.log")
    logger.remove()
    logger.add(log_path, level="INFO")
    return logger


def create_writer(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)