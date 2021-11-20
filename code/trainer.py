from torch.optim import Adam, AdamW
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from load_model import load_model


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
        self.model_setup(model_args)

        self.training_datset = training_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

    def model_setup(self, model_args):

        self.model = load_model(model_args)
        self.loss_fn = nn.CrossEntropyLoss()
        # TODO: more fancy way to check this out
        if model_args.model_name_or_path in ["bert"]:
            self.setup_transformer()

        elif model_args.model_name_or_path in ["cnn-lstm"]:
            self.setup_recurrent()

        else:
            self.optimizer = Adam(self.model)
            self.optimizer.zero_grad()

    def setup_transformer(self):

        if self.trainin_dataset is not None:
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
                len(self.train_dataloader)
                // self.training_args.gradient_accumulation_steps
                * self.training_args.num_train_epochs
            )
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.training_args.warmup_steps,
                num_training_steps=t_total,
            )

    def setup_recurrent(self):

        pass

    def run(self):

        pass

    def train(self, dataset=None):

        if dataset is None:
            dataset = self.training_dataset

        for batch in dataset:

            batch = {k: v.cuda() for k, v in batch.items()}
            logits, predicted_class = self.model(**batch)
            loss = self.loss_fn(logits, batch["labels"])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def valid(self, dataset=None):

        pass

    def save_state(self):

        pass


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
