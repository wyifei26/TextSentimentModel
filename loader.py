import numpy as np
import torch
from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from utils import timeit


class TextDataset(Dataset):
    def __init__(self, texts, labels, word_vectors, seq_len):
        self.texts = texts
        self.labels = labels
        assert len(self.texts) == len(self.labels)
        self.word_vectors = torch.tensor(word_vectors, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = []
        for word_id in self.texts[idx]:
            if word_id < len(self.word_vectors):
                text.append(self.word_vectors[word_id])
            else:
                text.append(torch.zeros(self.word_vectors.shape[1]))

        if len(text) < self.seq_len:
            text += [torch.zeros(self.word_vectors.shape[1])] * (self.seq_len - len(text))
        else:
            text = text[:self.seq_len]

        label = self.labels[idx]
        return torch.stack(text), torch.tensor(label, dtype=torch.long)  # 返回堆叠的张量


@timeit
def preprocess(seq_len):
    # word_dict为词典，key为词，value为词的id
    word_dict = {}
    datasets = []
    for path in ['./Dataset/test.txt', './Dataset/train.txt', './Dataset/validation.txt']:
        texts, labels = [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                # 以\t分割标签和文本
                label, text = line.strip().split('\t', 1)
                texts.append(text.split())
                labels.append(int(label))

        # 生成词典
        for text in texts:
            for word in text:
                if word not in word_dict:
                    word_dict[word] = len(word_dict)

        datasets.append((texts, labels))

    # 生成词向量
    word2vec = KeyedVectors.load_word2vec_format('./Dataset/wiki_word2vec_50.bin', binary=True)
    word_vectors = np.array(np.zeros([len(word_dict) + 1, word2vec.vector_size]))
    for word, id in word_dict.items():
        if word in word2vec:
            word_vectors[id] = word2vec[word]

    # 将datasets转化为word_id
    for i, (texts, labels) in enumerate(datasets):
        for j, text in enumerate(texts):
            for k, word in enumerate(text):
                text[k] = word_dict[word]
            datasets[i][0][j] = text

    # 创建TextDataset实例
    test_dataset = TextDataset(datasets[0][0], datasets[0][1], word_vectors, seq_len)
    train_dataset = TextDataset(datasets[1][0], datasets[1][1], word_vectors, seq_len)
    valid_dataset = TextDataset(datasets[2][0], datasets[2][1], word_vectors, seq_len)

    return word_dict, word_vectors, test_dataset, train_dataset, valid_dataset


def collate_batch(batch):
    label_list, text_list = [], []
    for _text, _label in batch:
        label_list.append(_label)
        text_list.append(torch.tensor(_text, dtype=torch.float))  # 确保文本数据是浮点类型
    text_tensor = pad_sequence(text_list, batch_first=True, padding_value=0)  # 使用batch_first=True
    label_tensor = torch.tensor(label_list, dtype=torch.long)
    return text_tensor, label_tensor

