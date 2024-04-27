import torch
from utils import vectorizeWord, enumerateWord

word2idx = enumerateWord()
word2vec = vectorizeWord(word2idx)


class Hyperparameter:
    update_w2v = True
    vocab_size = len(word2idx) + 1
    n_class = 2
    embedding_dim = 50
    dropout = 0.3
    kernel_num = 20
    kernel_size = [3, 5, 7]
    pretrained_embed = word2vec
    hidden_size = 100
    num_layers = 2


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
