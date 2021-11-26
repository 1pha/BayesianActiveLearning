import logging

import wandb
from tqdm import trange
from sklearn.metrics import accuracy_score, roc_auc_score

import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch.optim import Adam, AdamW
from transformers import get_linear_schedule_with_warmup
from utils.file_utils import save_state

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
        data_args,
        training_args,
        model_args,
        training_dataset,
        validation_dataset,
        test_dataset,
    ):

        self.data_args = data_args
        self.training_args = training_args
        self.model_args = model_args

        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

        self.model_setup(model_args)

        self.model_args.num_params = self.num_params

    @property
    def num_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def model_setup(self, model_args):

        self.model = load_model(model_args)
        if self.training_args.use_gpu:
            self.model = self.model.to("cuda")

        self.loss_fn = nn.BCEWithLogitsLoss()
        # TODO: more fancy way to check this out

        model_name = model_args.model_name_or_path
        if model_name in ["bert"]:
            logger.info(
                f"{model_name.capitalize()} was selected. Start Transformer setup."
            )
            self.setup_transformer()

        elif model_name in ["cnnlstm", "bilstm"]:
            logger.info(
                f"{model_name.capitalize()} was selected. Start Recurrent Networks setup."
            )
            self.setup_recurrent()

        else:
            logger.info(
                f"{model_name} was selected. Default configuration is used - naive Adam."
            )
            self.optimizer = Adam(self.model.parameters())
            self.optimizer.zero_grad()

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.training_args.fp16)

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

        self.optimizer = Adam(self.model.parameters())
        self.optimizer.zero_grad()
        logger.info("Successfully setup recurrent neural network settings.")

    def run(self):

        checkpoint_period = 0
        pbar = trange(self.training_args.num_train_epochs, desc="Epoch")
        for e in pbar:

            _, train_metrics = self.train(self.training_dataset)
            with torch.no_grad():
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

    def train(self, dataset=None):

        self.model.train()
        if dataset is None:
            dataset = self.training_dataset
            if dataset is None:
                # If do_train is set to False, there is no training set for loop
                return 0, self.get_metric()

        logits, labels, losses = [], [], []
        for batch in dataset:

            if self.training_args.use_gpu:
                batch = {k: v.cuda() for k, v in batch.items()}

            with torch.cuda.amp.autocast(self.training_args.fp16):
                logit, _ = self.model(batch)

                loss = self.loss_fn(
                    logit,
                    one_hot(
                        batch["labels"], num_classes=self.model_args.num_labels
                    ).type(torch.float),
                )
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if hasattr(self, "scheduler"):
                    self.scheduler.step()
                self.optimizer.zero_grad()

            logits.append(nn.Softmax(dim=1)(logit.detach().cpu().type(torch.float32)))
            labels.append(batch["labels"].cpu())
            losses.append(loss.item())

            torch.cuda.empty_cache()
            del batch, logit, loss

        loss = sum(losses) / len(losses)

        labels = torch.cat(labels).numpy()
        logits = torch.vstack(logits)
        metrics = self.get_metric(y_true=labels, y_pred=logits)

        wandb.log(
            {
                "train_loss(epoch)": loss,
                "train_acc(epoch)": metrics["acc"],
                "train_auroc(epoch)": metrics["auroc"],
            },
            commit=False,
        )

        return (loss, metrics)

    def valid(self, dataset=None, split="valid"):

        self.model.eval()
        if dataset is None:
            dataset = self.validation_dataset
            split = "valid"
            if dataset is None:
                # If do_valid is set to False, there is no training set for loop
                return 0, self.get_metric()

        logits, labels, losses = [], [], []
        for batch in dataset:

            if self.training_args.use_gpu:
                batch = {k: v.cuda() for k, v in batch.items()}
            logit, _ = self.model(batch)

            loss = self.loss_fn(
                logit,
                one_hot(batch["labels"], num_classes=self.model_args.num_labels).type(
                    torch.float
                ),
            )

            logits.append(nn.Softmax(dim=1)(logit).detach().cpu())
            labels.append(batch["labels"].cpu())
            losses.append(loss.item())

            torch.cuda.empty_cache()
            del batch, logit, loss

        loss = sum(losses) / len(losses)

        labels = torch.cat(labels).numpy()
        logits = torch.vstack(logits)
        metrics = self.get_metric(y_true=labels, y_pred=logits)

        wandb.log(
            {
                f"{split}_loss(epoch)": loss,
                f"{split}_acc(epoch)": metrics["acc"],
                f"{split}_auroc(epoch)": metrics["auroc"],
            },
            commit=False,
        )

        return (loss, metrics)

    def get_metric(self, y_true=None, y_pred=None):

        if y_true is None and y_pred is None:
            return {"acc": 0, "auroc": 0}
        else:
            return {
                "acc": accuracy_score(y_true, y_pred.argmax(axis=1)),
                "auroc": roc_auc_score(y_true, y_pred, multi_class="ovr"),
            }
