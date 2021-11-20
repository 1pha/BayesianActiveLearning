import logging

from config import parse_arguments
from dataset import build_dataset, build_dataloader

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("main.py")


def main():

    data_args, training_args, model_args = parse_arguments()
    logger.info(f"Start Training.")

    # 1. Initialize Dataset
    dataset = build_dataset(data_args, "train")
    dataloader = build_dataloader(dataset, data_args)
