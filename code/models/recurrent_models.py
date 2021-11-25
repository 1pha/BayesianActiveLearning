import torch
import torch.nn as nn
import torch.nn.functional as F


class baseRNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        input_dropout_p,
        output_dropout_p,
        n_layers,
        rnn_cell,
        max_len=25,
    ):
        super(baseRNN, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.max_len = max_len

        self.input_dropout_p = input_dropout_p
        self.output_dropout_p = output_dropout_p

        if rnn_cell.lower() == "lstm":
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == "gru":
            self.rnn_cell = nn.GRU
        else:
            raise ValueError(f"Unsupported RNN Cell: {rnn_cell}")

        self.input_dropout = nn.Dropout(p=input_dropout_p)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def num_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class EncoderRNN(baseRNN):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        hidden_size=200,
        input_dropout_p=0,
        output_dropout_p=0,
        n_layers=1,
        bidirectional=True,
        rnn_cell="lstm",
    ):

        super(EncoderRNN, self).__init__(
            vocab_size,
            hidden_size,
            input_dropout_p,
            output_dropout_p,
            n_layers,
            rnn_cell,
        )

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.rnn = self.rnn_cell(
            embedding_size,
            hidden_size,
            n_layers,
            bidirectional=bidirectional,
            dropout=output_dropout_p,
            batch_first=True,
        )

    def forward(self, words, input_lengths):

        batch_size = words.size()[0]
        embedded = self.embedding(words)
        embedded = self.input_dropout(embedded)
        # embedded = nn.utils.rnn.pack_padded_sequence(
        #     embedded, input_lengths, batch_first=True
        # )
        _, output = self.rnn(embedded)
        output = output[0].transpose(0, 1).contiguous().view(batch_size, -1)

        return output


class BiLSTM_MC(nn.Module):
    def __init__(self, config):

        super(BiLSTM_MC, self).__init__()

        self.word_vocab_size = config.vocab_size
        self.word_embedding_dim = config.embed_dim
        self.word_hidden_dim = config.intermediate_size
        self.n_layers = config.num_layers
        self.dropout_prob = config.dropout_prob
        self.bidirectional = config.bidirectional

        self.word_encoder = EncoderRNN(
            self.word_vocab_size,
            self.word_embedding_dim,
            self.word_hidden_dim,
            input_dropout_p=self.dropout_prob,
            output_dropout_p=self.dropout_prob,
            n_layers=self.n_layers,
            bidirectional=self.bidirectional,
            rnn_cell=config.rnn_cell,
        )
        self.dropout = nn.Dropout(p=config.dropout_prob)

        hidden_size = (
            2 * self.n_layers * self.word_hidden_dim
            if self.bidirectional
            else self.n_layers * self.word_hidden_dim
        )
        self.linear = nn.Linear(hidden_size, config.num_labels)
        self.lossfunc = nn.CrossEntropyLoss()

    def forward(self, data):

        input_ids = data["input_ids"]
        labels = data["labels"]
        wordslen = (input_ids == 2).nonzero()[:, 1]

        batch_size, max_len = input_ids.size()

        word_features = self.word_encoder(input_ids, wordslen)
        word_features = self.dropout(word_features)
        logits = self.linear(word_features)
        predicted_class = torch.argmax(logits, dim=1)

        return logits, predicted_class


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
    model = BiLSTM_MC(model_args)
    print(model(batch))
