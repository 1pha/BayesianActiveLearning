import json
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split
from datasets import load_from_disk
import datasets
from torch.utils.data.dataloader import DataLoader

from preprocess import build_preprocessor

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_dataset(config, split):

    return load_from_disk(Path(f"{config.data_dir}/{config.dataset_name}"))[split]


def build_dataset(config, split):

    full_dataset = load_dataset(config, split)
    dataset = initialize_dataset(full_dataset, config, split)

    preprocessor = build_preprocessor(config)
    vocab_size = len(preprocessor.tokenizer.vocab.strings)
    tokenize = preprocessor.batch_tokenize

    area2idx = load_area2idx(config)
    num_labels = len(area2idx)

    def batch_encode(example):

        input_raw = (
            example["title"]
            if "abstract" not in example
            else f"{example['title']}[SEP]{example['abstract']}"
        )
        input_ids = tokenize(input_raw)
        label = [area2idx[area] for area in example["area"]]

        return {
            "input_ids": input_ids,
            "labels": label,
        }

    remove_columns = ["area", "title"]
    remove_columns += ["abstract"] if "abstract" in dataset else []
    dataset = dataset.map(batch_encode, batched=True, remove_columns=remove_columns)
    logger.info(f"{split} dataset was properly preprocessed.")
    return dataset, vocab_size, num_labels


def initialize_dataset(dataset, config, split):

    """Get small portion from the whole for initial data.

    This will

    If init_pct receives 1 or the dataset is not for training,
    this will automatically return the intact input.

    Returns:
        [datasets] contains with iniitial percentage of the whole data
    """
    logger.info(f"Initialize {split.capitalize()} Dataset.")
    try:
        # When cache file is used, below codes might raise error.
        if not config.use_abstract:
            logger.info("Remove abstract.")
            dataset = dataset.remove_columns("abstract")

        if not config.use_task_id:
            logger.info("Remove task_id")
            dataset = dataset.remove_columns("task_id")

        dataset = dataset.remove_columns("arxiv_id")

    except:
        logger.info("Using cached dataset, wasn't able to remove columns.")
        pass

    if config.init_pct < 1 and split is "train":
        logger.info(f"Use {config.init_pct}% of the total dataset.")
        pool_dataset, initial_dataset = train_test_split(
            dataset, test_size=config.init_pct, shuffle=True, random_state=config.seed
        )
        pool_dataset = datasets.Dataset.from_dict(pool_dataset)
        initial_dataset = datasets.Dataset.from_dict(initial_dataset)
        logger.info(f"Total {len(initial_dataset)} of papers will be used.")

    else:
        logger.info(
            f"Use the full dataset, for {split} dataset of total {len(dataset)} papers."
        )
        initial_dataset = dataset

    logger.info(f"{split.capitalize()} dataset was successfully initialized.")
    return initial_dataset


def load_area2idx(config):

    fname = Path(f"{config.asset_dir}/{config.area2idx}")
    try:
        with open(fname) as f:
            area2idx = json.load(f)
        logger.info(f"Successfully loaded mapper file {fname}")
        return area2idx

    except:
        logger.warn(f"Failed to load mapper file {fname} ")
        raise


def build_dataloader(dataset, config):

    # Preprocessed dataset
    try:
        dataset.set_format(type="torch")
        dataloader = DataLoader(
            dataset, batch_size=config.batch_size, pin_memory=config.pin_memory
        )
        logger.info("Successfully converted dataset to dataloader.")
        return dataloader
    except:
        logger.warn("Failed to convert dataset into dataloader.")


if __name__ == "__main__":

    from config import parse_arguments

    data_args, training_args, model_args = parse_arguments()

    print(build_dataset(data_args, "train"))
