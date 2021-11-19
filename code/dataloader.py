from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from datasets import load_from_disk
import datasets

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

        try:
            # When cache file is used, below codes might raise error.
            self.full_dataset = self.full_dataset.remove_columns("arxiv_id")
            if not config.use_abstract:
                self.full_dataset.remove_columns("abstract")

            if not config.use_task_id:
                self.full_dataset.remove_columns("task_id")
        except:
            pass

        self.dataset = self.initialize_dataset(self.full_dataset, config.init_pct, split)


    def __getitem__(self, idx):

        # TODO preprocessing! - tokenize and map areas

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

        if init_pct < 1 and split is "train":
            pool_dataset, initial_dataset = train_test_split(
                dataset, test_size=init_pct, shuffle=True, random_state=self.config.seed
            )
            self.pool_dataset = datasets.Dataset.from_dict(pool_dataset)
            initial_dataset = datasets.Dataset.from_dict(initial_dataset)

            return initial_dataset

        else:
            setattr(self, "pool_dataset", None)
            return dataset

    def __repr__(self):
        return repr(self.dataset)


if __name__ == "__main__":

    from config import parse_arguments

    data_args, training_args = parse_arguments()
    dataset = PaperDataset(config=data_args)
    print(dataset)
