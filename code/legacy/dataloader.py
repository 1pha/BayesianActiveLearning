import json
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import datasets
from torch.utils.data.dataloader import DataLoader

from preprocess import Preprocessor

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class PaperDataset(Dataset):
    def __init__(self, config, split="train"):
        """Initialize dataset

        Args:
            config ([type]): [description]
            split (str, optional): [description]. Defaults to 'train'.
        """

        self.config = config
        self.full_dataset = load_from_disk(
            Path(f"{config.data_dir}/{config.dataset_name}")
        )[split]

        self.dataset = self.initialize_dataset(
            self.full_dataset, config.init_pct, split
        )
        self.setup_preprocessor(config)
        self.load_area2idx(config)
        logger.info(f"{split.capitalize()} dataset was successfully loaded.")

    def __getitem__(self, idx):

        data = self.dataset[idx]

        input_ids = self.tokenize(data["title"]).squeeze()
        if "abstract" in data:
            abstract_token = self.tokenize(data["abstract"]).squeeze()
            input_ids = torch.concat((input_ids, abstract_token))

        label = torch.tensor(self.area2idx[data["area"]], dtype=torch.long)

        seq_len = len(input_ids)
        if seq_len > self.config.max_seq_len:
            input_ids = input_ids[: self.config.max_seq_len]
            seq_len = self.config.max_seq_len

        else:
            zero_seq = torch.zeros(self.config.max_seq_len, dtype=torch.long)
            zero_seq[:seq_len] = input_ids
            input_ids, zero_seq = zero_seq, input_ids

        return {
            "input_ids": input_ids,
            "label": label,
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
        }

    def __len__(self):

        return len(self.dataset)

    def balance(self):

        # TODO make a method to balance the labels

        pass

    def initialize_dataset(self, dataset, init_pct, split):

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
            if not self.config.use_abstract:
                logger.info("Remove abstract.")
                dataset = dataset.remove_columns("abstract")

            if not self.config.use_task_id:
                logger.info("Remove task_id")
                dataset = dataset.remove_columns("task_id")

            dataset = dataset.remove_columns("arxiv_id")

        except:
            logger.info("Using cached dataset, wasn't able to remove columns.")
            pass

        if init_pct < 1 and split is "train":
            logger.info(f"Use {init_pct}% of the total dataset.")
            pool_dataset, initial_dataset = train_test_split(
                dataset, test_size=init_pct, shuffle=True, random_state=self.config.seed
            )
            self.pool_dataset = datasets.Dataset.from_dict(pool_dataset)
            initial_dataset = datasets.Dataset.from_dict(initial_dataset)
            logger.info(f"Total {len(initial_dataset)} of papers will be used.")

            return initial_dataset

        else:
            logger.info(
                f"Use the full dataset, for {split} dataset of total {len(dataset)} papers."
            )
            setattr(self, "pool_dataset", None)
            return dataset

    def setup_preprocessor(self, config):

        self.preprocessor = Preprocessor(config)
        self.tokenize = self.preprocessor.tokenize

    def load_area2idx(self, config):

        fname = Path(f"{config.asset_dir}/{config.area2idx}")
        try:
            with open(fname) as f:
                self.area2idx = json.load(f)
            logger.info(f"Successfully loaded mapper file {fname}")

        except:
            logger.warn(f"Failed to load mapper file {fname} ")
            raise

    def __repr__(self):
        return repr(self.dataset)


def collate_fn(batch):

    # return {
    #     "input_ids": torch.stack([b["input_ids"] for b in batch]),
    #     "labels": torch.stack([b["label"] for b in batch]),
    #     "seq_len": torch.stack([b["seq_len"] for b in batch]),
    # }
    data = {
        "input_ids": [],
        "labels": [],
        "seq_len": [],
    }
    for b in batch:
        data["input_ids"].append(b["input_ids"])
        data["labels"].append(b["label"])
        data["seq_len"].append(b["seq_len"])

    return {k: torch.stack(v) for k, v in data.items()}


def build_dataloader(config, split):

    dataset = PaperDataset(config, split)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )

    return dataloader


if __name__ == "__main__":

    from config import parse_arguments

    data_args, training_args, model_args = parse_arguments()
    # dataset = PaperDataset(config=data_args)
    # print(dataset)

    print(build_dataloader(data_args))
