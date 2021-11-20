import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification


class BaseTransformer(nn.Module):
    def __init__(self, config):

        self.model = BertForSequenceClassification()

    def forward(self, data):

        input_ids = data["input_ids"]
        labels = data["label"]

        logits = self.model(input_ids=input_ids, labels=labels)["logits"]
        predicted_class = torch.argmax(logits, dim=1)
        return out
