import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        update_w2v = config.update_w2v
        vocab_size = config.vocab_size
        n_class = config.n_class
        embedding_dim = config.embedding_dim
        kernel_num = config.kernel_num
        kernel_size = config.kernel_size
        drop_keep_prob = config.drop_keep_prob
        pretrained_embed = config.pretrained_embed

        self.__name__ = 'CNN'
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.requires_grad = update_w2v
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))

        self.conv1 = nn.Conv2d(1, kernel_num, (kernel_size[0], embedding_dim))
        self.conv2 = nn.Conv2d(1, kernel_num, (kernel_size[1], embedding_dim))
        self.conv3 = nn.Conv2d(1, kernel_num, (kernel_size[2], embedding_dim))

        self.dropout = nn.Dropout(drop_keep_prob)

        self.fc = nn.Linear(len(kernel_size) * kernel_num, n_class)

    @staticmethod
    def apply_conv_and_pool(input_tensor, conv_layer):
        conv_output = conv_layer(input_tensor).squeeze(3)
        conv_output = F.relu(conv_output)
        pooled_output = F.max_pool1d(conv_output, conv_output.size(2))
        return pooled_output.squeeze(2)

    def forward(self, input_sequence):
        embedded = self.embedding(input_sequence.to(torch.int64)).unsqueeze(1)

        conv1_output = self.apply_conv_and_pool(embedded, self.conv1)
        conv2_output = self.apply_conv_and_pool(embedded, self.conv2)
        conv3_output = self.apply_conv_and_pool(embedded, self.conv3)

        concatenated = torch.cat((conv1_output, conv2_output, conv3_output), dim=1)
        dropped = self.dropout(concatenated)

        output = self.fc(dropped)
        log_probs = F.log_softmax(output, dim=1)

        return log_probs


class RNN_LSTM(nn.Module):
    def __init__(self, config):
        super(RNN_LSTM, self).__init__()

        vocab_size = config.vocab_size
        update_w2v = config.update_w2v
        embedding_dim = config.embedding_dim
        pretrained_embed = config.pretrained_embed
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.n_class = config.n_class
        self.__name__ = 'RNN_LSTM'

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # ! embedding is a table, which is used to lookup the embedding vector of a word
        self.embedding.weight.requires_grad = update_w2v
        # ! if update_w2v is True, the embedding.weight will be updated during training
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))
        # ! import the pretrained embedding vector as embedding.weight

        # (seq_len, batch, embed_dim) -> (seq_len, batch, 2 * hidden_size)
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
        )
        # (batch, hidden_size * 2) -> (batch, num_classes)
        self.decoder = nn.Linear(2 * self.hidden_size, 64)
        self.fc1 = nn.Linear(64, self.n_class)
        # (batch, num_classes) -> (batch, num_classes)

    def forward(self, inputs):
        _, (h_n, _) = self.encoder(
            self.embedding(inputs.to(torch.int64)).permute(1, 0, 2))  # (num_layers * 2, batch, hidden_size)
        # view h_n as (num_layers, num_directions, batch, hidden_size)
        h_n = h_n.view(self.num_layers, 2, -1, self.hidden_size)
        return self.fc1(self.decoder(torch.cat((h_n[-1, 0], h_n[-1, 1]), dim=-1)))


class RNN_GRU(nn.Module):
    def __init__(self, config):
        super(RNN_GRU, self).__init__()

        vocab_size = config.vocab_size
        update_w2v = config.update_w2v
        embedding_dim = config.embedding_dim
        pretrained_embed = config.pretrained_embed
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.n_class = config.n_class
        self.__name__ = 'RNN_GRU'

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # ! embedding is a table, which is used to lookup the embedding vector of a word
        self.embedding.weight.requires_grad = update_w2v
        # ! if update_w2v is True, the embedding.weight will be updated during training
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))
        # ! import the pretrained embedding vector as embedding.weight

        # (seq_len, batch, embed_dim) -> (seq_len, batch, 2 * hidden_size)
        self.encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
        )
        # (batch, hidden_size * 2) -> (batch, num_classes)
        self.decoder = nn.Linear(2 * self.hidden_size, 64)
        self.fc = nn.Linear(64, self.n_class)
        # (batch, num_classes) -> (batch, num_classes)

    def forward(self, inputs):
        x = self.embedding(inputs.to(torch.int64)).permute(1, 0, 2)
        h_0 = torch.rand(self.num_layers * 2, x.size(1), self.hidden_size).to(
            torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        _, h_n = self.encoder(x, h_0)  # (num_layers * 2, batch, hidden_size)
        h_n = h_n.view(self.num_layers, 2, -1,
                       self.hidden_size)  # view h_n as (num_layers, num_directions, batch, hidden_size)
        return self.fc(self.decoder(torch.cat((h_n[-1, 0], h_n[-1, 1]), dim=-1)))


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        vocab_size = config.vocab_size
        update_w2v = config.update_w2v
        embedding_dim = config.embedding_dim
        pretrained_embed = config.pretrained_embed
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.n_class = config.n_class
        self.__name__ = 'MLP'
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # ! embedding is a table, which is used to lookup the embedding vector of a word
        self.embedding.weight.requires_grad = update_w2v
        # ! if update_w2v is True, the embedding.weight will be updated during training
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))
        # ! import the pretrained embedding vector as embedding.weight
        self.relu = torch.nn.ReLU()
        self.mlp_layer = torch.nn.Linear(embedding_dim, self.hidden_size)
        self.linear = torch.nn.Linear(self.hidden_size, self.n_class)
        # init weights
        for _, p in self.named_parameters():
            if p.requires_grad:
                torch.nn.init.normal_(p, mean=0, std=0.01)

    def forward(self, inputs):
        output = self.relu(self.mlp_layer(self.embedding(inputs.to(torch.int64)))).permute(0, 2, 1)  # B * h * len
        return self.linear(F.max_pool1d(output, output.shape[2]).squeeze(2))
