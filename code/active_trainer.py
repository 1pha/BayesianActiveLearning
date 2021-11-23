import logging

import wandb
from tqdm import trange

import torch
import torch.nn as nn
import datasets

from trainer import NaiveTrainer
from dataset import build_dataloader
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

        if config.acquisition in ["random", "lc", "mc", "entropy"]:
            config.approximation = "single"
            config.num_sampling = 1

        return config

    def run(self, training_dataset=None, validation_dataset=None, test_dataset=None):

        checkpoint_period, acquisition_period = 0, 0
        pbar = trange(self.training_args.num_train_epochs, desc="Epoch")
        for e in pbar:

            train_loss, train_metrics = self.train(training_dataset)
            valid_loss, valid_metrics = self.valid(
                validation_dataset=validation_dataset
            )
            test_loss, test_metrics = self.valid(test_dataset=test_dataset)

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
            if self.training_args.acquisition_period:

                acquire_data, self.pool_dataset = self.acquire_batch(
                    pool_dataset=self.pool_dataset
                )

                training_dataset = build_dataloader(
                    datasets.concatenate_datasets(
                        (training_dataset.dataset, acquire_data)
                    )
                )
                acquisition_period = 0

    def acquire_batch(self, pool_dataset=None):

        if pool_dataset is None:
            logger.warn(f"No dataset given. Please give pool data.")

        logits = self.retrieve_logit(pool_dataset)
        confidence_level = self.calculate_information(logits)

        idx = confidence_level.argsort()

        num_acquire = self.training_args.increment_num
        acquired_idx = idx[num_acquire:]
        leftover_idx = idx[:num_acquire]

        acquired_data = pool_dataset[acquired_idx]
        pool_dataset = pool_dataset[leftover_idx]

        return acquired_data, pool_dataset

    def retrieve_logit(self, dataset):

        self.model.training = (
            True if self.training_args.approximation == "mcdropout" else False
        )

        logit_models = []
        for k in range(self.num_samplings):

            logits = []
            for batch in dataset:
                if self.training_args.use_gpu:
                    batch = {k: v.cuda() for k, v in batch.items()}
                logit, _ = self.model(batch)

                logits.append(nn.Softmax(dim=1)(logit).detach().cpu())

                torch.cuda.empty_cache()
                del batch, logit

            logits = torch.vstack(logits)
            logit_models.append(logits)

        return logit_models
