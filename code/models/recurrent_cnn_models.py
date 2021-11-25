import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, out_channels=100, dropout_p=0):

        super(EncoderCNN, self).__init__()

        self.out_channels = out_channels
        self.dropout = nn.Dropout(p=dropout_p)

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        in_channels = embedding_size
        self.cnn1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(out_channels, out_channels, kernel_size=4, padding=1)
        self.cnn3 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=1)

    def forward(self, words, input_lengths=None):

        batch_size, _ = words.size()

        embedded = self.embedding(words)
        embedded = self.dropout(embedded)
        embedded = embedded.transpose(1, 2)
        output1 = F.relu(self.cnn1(embedded))
        output2 = F.relu(self.cnn2(output1))
        output3 = F.relu(self.cnn3(output2))
        output = nn.functional.max_pool1d(output3, kernel_size=output3.size(2))
        output = output.squeeze(2)

        return output

    @property
    def num_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class CNN_MC(nn.Module):
    def __init__(self, config):

        super(CNN_MC, self).__init__()

        self.word_vocab_size = config.vocab_size
        self.word_embedding_dim = config.embed_dim
        self.word_hidden_dim = config.intermediate_size
        self.dropout_prob = config.dropout_prob
        self.word_out_channels = config.out_channels

        self.word_encoder = EncoderCNN(
            self.word_vocab_size, self.word_embedding_dim, self.word_out_channels
        )

        self.dropout = nn.Dropout(p=self.dropout_prob)

        hidden_size = self.word_out_channels
        self.linear = nn.Linear(hidden_size, config.num_labels)

        self.lossfunc = nn.CrossEntropyLoss()

    def forward(self, data):

        input_ids = data["input_ids"]
        labels = data["labels"]

        batch_size, max_len = input_ids.size()
        word_features = self.word_encoder(input_ids)
        word_features = self.dropout(word_features)
        output = self.linear(word_features)

        return output


if __name__ == "__main__":

    import sys

    sys.path.append("../")
    from config import parse_arguments
    from dataset import build_dataset, build_dataloader

    data_args, training_args, model_args = parse_arguments()
    data_args.data_dir = "../../data"
    data_args.asset_dir = "../../assets"

    (
        pool_dataset,
        train_dataset,
        model_args.vocab_size,
        model_args.num_labels,
    ) = build_dataset(data_args, "train")
    train_dataloader = build_dataloader(train_dataset, data_args)

    batch = next(iter(train_dataloader))
    model = CNN_MC(model_args)
    print(model(batch).shape)
