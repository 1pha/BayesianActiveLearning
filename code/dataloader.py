from torch.utils.data import Dataset, DataLoaders
from datasets import load_from_disk


class PaperDataset(Dataset):
    def __init__(
        self,
        config,
    ):

        self.config = config
        self.dataset = load_from_disk(f"{config.data_dir}/{config.dataset_name}")

    def __getitem__(self, idx):

        return

    def __len__(self):

        return

    def balance(self):

        pass

    def add_data(self, pct):

        pass
