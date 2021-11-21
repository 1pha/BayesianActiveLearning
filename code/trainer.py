import logging

from sklearn.metrics import accuracy_score, roc_auc_score

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from transformers import get_linear_schedule_with_warmup

from load_model import load_model


logging.basicConfig(
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class NaiveTrainer:
    def __init__(
        self,
        training_args,
        model_args,
        training_dataset,
        validation_dataset,
        test_dataset,
    ):

        self.training_args = training_args
        self.model_args = model_args

        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

        self.model_setup(model_args)

    def model_setup(self, model_args):

        self.model = load_model(model_args)
        if self.training_args.use_gpu:
            self.model = self.model.to("cuda")

        self.loss_fn = nn.CrossEntropyLoss()
        # TODO: more fancy way to check this out

        model_name = model_args.model_name_or_path
        if model_name in ["bert"]:
            logger.info(
                f"{model_name.capitalize()} was selected. Start Transformer setup."
            )
            self.setup_transformer()

        elif model_name in ["cnn-lstm"]:
            logger.info(
                f"{model_name.capitalize()} was selected. Start Recurrent Networks setup."
            )
            self.setup_recurrent()

        else:
            logger.info(
                f"{model_name} was selected. Default configuration is used - naive Adam."
            )
            self.optimizer = Adam(self.model)
            self.optimizer.zero_grad()

    def setup_transformer(self):

        try:
            if self.training_dataset is not None:
                no_decay = ["bias", "LayerNorm.weight"]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in self.model.named_parameters()
                            if not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": self.training_args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in self.model.named_parameters()
                            if any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
                self.optimizer = AdamW(
                    optimizer_grouped_parameters, lr=self.training_args.learning_rate
                )
                self.optimizer.zero_grad()

                t_total = (
                    len(self.training_dataset)
                    // self.training_args.gradient_accumulation_steps
                    * self.training_args.num_train_epochs
                )
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=self.training_args.warmup_steps,
                    num_training_steps=t_total,
                )
                logger.info("Successfully setup transformer settings.")
            else:
                logger.info(
                    "No training dataset was given. Halt configuring transformer setup."
                )

        except:
            logger.warn("Failed to setup transformer settings.")
            raise

    def setup_recurrent(self):

        pass

    def run(self):

        pass

    def train(self, dataset=None):

        if dataset is None:
            dataset = self.training_dataset
            if dataset is None:
                # If do_train is set to False, there is not training set for loop
                return 0, self.get_metric()

        predictions, labels, losses = [], [], []
        for batch in dataset:

            if self.training_args.use_gpu:
                batch = {k: v.cuda() for k, v in batch.items()}
            logits, predicted_class = self.model(batch)

            loss = self.loss_fn(logits, batch["labels"])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            predictions.extend(predicted_class.cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
            losses.append(loss.items())

        loss = sum(losses) / len(losses)
        metrics = self.get_metric(labels, predictions)
        torch.cuda.empty_cache()

        return (loss, metrics)

    def valid(self, dataset=None):

        pass

    def save_state(self):

        pass

    def get_metric(self, true=None, pred=None):

        if true is None and pred is None:
            return {"acc": 0, "auroc": 0}
        else:
            return {
                "acc": accuracy_score(true, pred),
                "auroc": roc_auc_score(true, pred),
            }


class ActiveTrainer(NaiveTrainer):
    def __init__(
        self,
        training_args,
        model_args,
        training_dataset,
        validation_dataset,
        test_dataset,
    ):
        super().__init__(
            training_args,
            model_args,
            training_dataset,
            validation_dataset,
            test_dataset,
        )
