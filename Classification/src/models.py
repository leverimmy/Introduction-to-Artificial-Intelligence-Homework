import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertConfig

from para import DEVICE


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()

        self.__name__ = 'CNN'

        update_w2v = config.update_w2v
        vocab_size = config.vocab_size
        n_class = config.n_class
        embedding_dim = config.embedding_dim
        kernel_num = config.kernel_num
        kernel_size = config.kernel_size
        dropout = config.dropout
        pretrained_embed = config.pretrained_embed

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.requires_grad = update_w2v
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))

        self.conv1 = nn.Conv2d(1, kernel_num, (kernel_size[0], embedding_dim))
        self.conv2 = nn.Conv2d(1, kernel_num, (kernel_size[1], embedding_dim))
        self.conv3 = nn.Conv2d(1, kernel_num, (kernel_size[2], embedding_dim))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_size) * kernel_num, n_class)

    @staticmethod
    def conv_and_pool(x, conv):
        x = F.relu(conv(x).squeeze(3))
        return F.max_pool1d(x, x.size(2)).squeeze(2)

    def forward(self, inputs):
        x = self.embedding(inputs.to(torch.int64)).unsqueeze(1)
        x1 = self.conv_and_pool(x, self.conv1)
        x2 = self.conv_and_pool(x, self.conv2)
        x3 = self.conv_and_pool(x, self.conv3)
        return F.log_softmax(self.fc(self.dropout(torch.cat((x1, x2, x3), 1))), dim=1)


class RNN_LSTM(nn.Module):
    def __init__(self, config):
        super(RNN_LSTM, self).__init__()

        self.__name__ = 'RNN_LSTM'

        update_w2v = config.update_w2v
        vocab_size = config.vocab_size
        embedding_dim = config.embedding_dim
        pretrained_embed = config.pretrained_embed

        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.n_class = config.n_class
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.requires_grad = update_w2v
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))

        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
        )
        self.decoder = nn.Linear(2 * self.hidden_size, 64)
        self.fc = nn.Linear(64, self.n_class)

    def forward(self, inputs):
        _, (h_n, _) = self.encoder(
            self.embedding(inputs.to(torch.int64)).permute(1, 0, 2))

        h_n = h_n.view(self.num_layers, 2, -1, self.hidden_size)
        return self.fc(self.decoder(torch.cat((h_n[-1, 0], h_n[-1, 1]), dim=-1)))


class RNN_GRU(nn.Module):
    def __init__(self, config):
        super(RNN_GRU, self).__init__()

        self.__name__ = 'RNN_GRU'

        vocab_size = config.vocab_size
        update_w2v = config.update_w2v
        embedding_dim = config.embedding_dim
        pretrained_embed = config.pretrained_embed
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.n_class = config.n_class

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.requires_grad = update_w2v
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))

        self.encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
        )
        self.decoder = nn.Linear(2 * self.hidden_size, 64)
        self.fc = nn.Linear(64, self.n_class)

    def forward(self, inputs):
        x = self.embedding(inputs.to(torch.int64)).permute(1, 0, 2)
        h_0 = torch.rand(self.num_layers * 2, x.size(1), self.hidden_size).to(DEVICE)
        _, h_n = self.encoder(x, h_0)
        h_n = h_n.view(self.num_layers, 2, -1, self.hidden_size)
        return self.fc(self.decoder(torch.cat((h_n[-1, 0], h_n[-1, 1]), dim=-1)))


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()

        self.__name__ = 'MLP'

        vocab_size = config.vocab_size
        update_w2v = config.update_w2v
        embedding_dim = config.embedding_dim
        pretrained_embed = config.pretrained_embed
        
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.n_class = config.n_class

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.requires_grad = update_w2v
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))

        self.relu = torch.nn.ReLU()
        self.mlp_layer = torch.nn.Linear(embedding_dim, self.hidden_size)
        self.linear = torch.nn.Linear(self.hidden_size, self.n_class)

        for _, p in self.named_parameters():
            if p.requires_grad:
                torch.nn.init.normal_(p, mean=0, std=0.01)

    def forward(self, inputs):
        output = self.relu(self.mlp_layer(self.embedding(inputs.to(torch.int64)))).permute(0, 2, 1)
        return self.linear(F.max_pool1d(output, output.shape[2]).squeeze(2))


class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.__name__ = 'BERT'

        bert_config = BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.hidden_size // 64,  # 根据论文设置
            intermediate_size=config.hidden_size * 4,  # 根据论文设置
            max_position_embeddings=config.embedding_dim  # 根据最大序列长度设置
        )

        self.embedding = BertEmbeddings(bert_config)
        self.encoder = BertEncoder(bert_config)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_size, config.n_class)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        encoded = self.encoder(embedded)[0]  # 取最后一层的输出
        pooled = encoded[:, 0]  # 取 [CLS] token
        after_dropout = self.dropout(pooled)
        results = self.fc(after_dropout)
        return results
