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
    def __init__(
        self,
        word_vocab_size,
        word_embedding_dim,
        word_out_channels,
        output_size,
        dropout_p=0.5,
        pretrained=None,
    ):

        super(CNN_MC, self).__init__()

        self.word_vocab_size = word_vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.word_out_channels = word_out_channels

        self.word_encoder = EncoderCNN(
            word_vocab_size, word_embedding_dim, word_out_channels
        )

        if pretrained is not None:
            self.word_encoder.embedding.weight = nn.Parameter(
                torch.FloatTensor(pretrained)
            )

        self.dropout = nn.Dropout(p=dropout_p)

        hidden_size = word_out_channels
        self.linear = nn.Linear(hidden_size, output_size)

        self.lossfunc = nn.CrossEntropyLoss()

    def forward(self, words, tags, wordslen, usecuda=True):

        batch_size, max_len = words.size()
        word_features = self.word_encoder(words, wordslen)
        word_features = self.dropout(word_features)
        output = self.linear(word_features)
        loss = self.lossfunc(output, tags)

        return loss

    def predict(self, words, wordslen, scoreonly=False, usecuda=True):

        batch_size, max_len = words.size()
        word_features = self.word_encoder(words, wordslen)
        word_features = self.dropout(word_features)
        output = self.linear(word_features)

        scores = torch.max(F.softmax(output, dim=1), dim=1)[0].data.cpu().numpy()
        if scoreonly:
            return scores

        prediction = torch.max(output, dim=1)[1].data.cpu().numpy().tolist()
        return scores, prediction
