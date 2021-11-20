import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification


class BaseTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, data):

        input_ids = data["input_ids"]
        labels = data["labels"]

        logits = self.model(input_ids=input_ids, labels=labels)["logits"]
        predicted_class = torch.argmax(logits, dim=1)
        return logits, predicted_class


class Bert(BaseTransformer):
    def __init__(self, config):
        super().__init__(config)
        self.config = BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.embed_dim,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_attention_heads,
            hidden_dropout_prob=config.dropout_prob,
            attention_probs_dropout_prob=config.dropout_prob,
            classifier_dropout=config.dropout_prob,
            num_labels=config.num_labels,
        )
        self.model = BertForSequenceClassification(self.config)
