from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import datasets

from preprocess import Preprocessor

import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
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

    def __getitem__(self, idx):

        # TODO preprocessing! - tokenize and map areas
        data = self.dataset[idx]

        title_token = self.preprocessor.tokenize(data["title"])

        return self.dataset[idx]

    def __len__(self):

        return len(self.dataset)

    def balance(self):

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

    def __repr__(self):
        return repr(self.dataset)


if __name__ == "__main__":

    from config import parse_arguments

    data_args, training_args = parse_arguments()
    dataset = PaperDataset(config=data_args)
    print(dataset)
