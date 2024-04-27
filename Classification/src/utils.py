import gensim
import numpy as np


def enumerateWord():
    cnt = 0
    word2idx = {}
    for filename in ['train.txt', 'validation.txt']:
        with open(f'../Dataset/{filename}', 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()[1:]
                for word in words:
                    if word not in word2idx:
                        word2idx[word] = cnt
                        cnt += 1
    return word2idx


def vectorizeWord(word2idx):
    wiki = gensim.models.KeyedVectors.load_word2vec_format('../Dataset/wiki_word2vec_50.bin', binary=True)
    word2vec = np.array(np.zeros([len(word2idx) + 1, wiki.vector_size]))
    for key in word2idx:
        try:
            word2vec[word2idx[key]] = wiki[key]
        except Exception:
            pass
    return word2vec


def load_data(file_name, max_length, word2idx):
    contents, labels = np.array([0] * max_length), np.array([])

    with open(file_name, 'r', encoding="utf-8") as f:
        for line in f:
            sentence = line.strip().split()
            label = sentence[0]
            words = sentence[1:]

            content_raw = np.asarray([word2idx.get(word, 0) for word in words])[:max_length]
            padding = max(max_length - len(content_raw), 0)
            content = np.pad(content_raw, (0, padding), "constant", constant_values=0)

            contents = np.vstack([contents, content])
            labels = np.append(labels, int(label))

    contents = np.delete(contents, 0, axis=0)
    return contents, labels
