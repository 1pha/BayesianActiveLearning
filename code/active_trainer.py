import logging
import json
from pathlib import Path

import wandb
from tqdm import trange, tqdm

import torch
import torch.nn as nn
import datasets

from trainer import NaiveTrainer
from dataset import build_dataloader
from acquisition_function import AcquisitionTool
from utils.file_utils import save_state


logging.basicConfig(
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class ActiveTrainer(NaiveTrainer):
    def __init__(
        self,
        data_args,
        training_args,
        model_args,
        training_dataset,
        pool_dataset,
        validation_dataset,
        test_dataset,
    ):
        super().__init__(
            data_args,
            training_args,
            model_args,
            training_dataset,
            validation_dataset,
            test_dataset,
        )
        self.pool_dataset = pool_dataset
        assert pool_dataset is not None

        self.training_args = self.setup_active_learning(training_args)

    def setup_active_learning(self, config):

        """
        This function setup few configurations regarding the dependency between arguments.

        Priority
            Acquisition function is more prioritized than approximation method.
            If least confidence and dropout is chosen, dropout will automatically be removed.
        """

        self.num_sampling = config.num_sampling
        self.acquisition = AcquisitionTool(self.data_args, self.training_args)
        self.confidence_level_all = dict()

        return config

    def run(self):

        checkpoint_period, acquisition_period = 0, 0
        pbar = trange(self.training_args.num_train_epochs, desc="Epoch")
        for e in pbar:

            _, train_metrics = self.train(self.training_dataset)
            _, valid_metrics = self.valid(self.validation_dataset, split="valid")
            _, test_metrics = self.valid(self.test_dataset, split="test")

            wandb.log({"epoch": e}, commit=True)

            postfix = (
                test_metrics["auroc"]
                if test_metrics["auroc"] == 0
                else valid_metrics["auroc"]
            )
            postfix = train_metrics["auroc"] if postfix == 0 else postfix
            pbar.set_postfix(AUROC=f"{postfix:4f}")

            checkpoint_period += 1
            if checkpoint_period % self.training_args.checkpoint_period == 0:
                save_state(self.model, e, self.training_args)
                checkpoint_period = 0
            torch.cuda.empty_cache()

            acquisition_period += 1
            if acquisition_period % self.training_args.acquisition_period == 0:

                logger.info(
                    f"Start acquiring data with {self.acquisition.name} method."
                )
                acquired_data, self.pool_dataset = self.acquire_batch(
                    pool_dataset=self.pool_dataset, epoch=e
                )

                self.training_dataset = build_dataloader(
                    datasets.concatenate_datasets(
                        (self.training_dataset.dataset, acquired_data)
                    ),
                    self.data_args,
                )
                logger.info(
                    f"Acquisition done. Now use {len(self.training_dataset.dataset)} number of data."
                )
                with open(
                    Path(f"{self.training_args.output_dir}/confidence_level.json"), "w"
                ) as f:
                    json.dump(self.confidence_level_all, f)
                acquisition_period = 0

                if self.training_args.retrain:
                    self.model_setup(self.model_args)

    def acquire_batch(self, pool_dataset=None, epoch=None):

        if pool_dataset is None:
            logger.warn(f"No dataset given. Please give pool data.")
            raise

        if self.acquisition.method == "random":
            idx = self.acquisition(len(pool_dataset))

        else:
            confidence_level, labels = self.retrieve_confidence(pool_dataset)
            if self.training_args.save_confidence:
                self.confidence_level_all[epoch] = (labels, confidence_level.tolist())

            idx = confidence_level.argsort().tolist()

        num_acquire = self.training_args.increment_num
        acquired_idx = idx[:num_acquire]
        leftover_idx = idx[num_acquire:]

        acquired_data = datasets.Dataset.from_dict(pool_dataset.dataset[acquired_idx])
        pool_dataset = build_dataloader(
            datasets.Dataset.from_dict(pool_dataset.dataset[leftover_idx]),
            self.data_args,
        )

        return acquired_data, pool_dataset

    def retrieve_confidence(self, dataset):

        self.model.training = (
            True if self.training_args.approximation == "mcdropout" else False
        )

        confidence_levels = []  # ((num_model, class), ) * batch
        labels = []
        pbar = tqdm(dataset, desc="Pool Dataset")
        for batch in pbar:

            if self.training_args.use_gpu:
                batch = {k: v.cuda() for k, v in batch.items()}

            logits = []
            for k in range(self.num_sampling):

                logit, _ = self.model(batch)
                logits.append(nn.Softmax(dim=1)(logit).detach())
                torch.cuda.empty_cache()
                del logit

            logits = torch.vstack(logits)
            labels.append(batch["labels"].cpu())
            confidence = self.acquisition(logits).cpu()
            del batch

            confidence_levels.append(confidence)

        labels = torch.cat(labels).numpy()
        confidence_levels = torch.cat(confidence_levels)

        return confidence_levels, labels
