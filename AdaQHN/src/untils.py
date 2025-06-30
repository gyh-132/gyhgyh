import torch
import torch.nn as nn
import collections
import re
import sys
import string
import numpy as np
import random
from pathlib import Path


class RMSELoss(nn.Module):
    """均方误差损失函数"""
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        mse_loss = torch.mean((y_pred - y_true) ** 2)  # 计算均方误差
        rmse_loss = torch.sqrt(mse_loss)  # 计算均方根误差
        return rmse_loss


def read_txt(path):
    """提取文本到列表"""
    with open(path, 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)


class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        if len(tokens) == 0 or isinstance(tokens[0], list):
            # 将词元列表展平成一个列表
            tokens = [token for line in tokens for token in line]
        else:
            print("Vocab chu cuo!")
            sys.exit()  # 终止程序
        # 按出现频率排序
        counter = collections.Counter(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, 0)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    def token_freqs(self):
        return self._token_freqs


def load_corpus_vocab(path, token='word', max_tokens=-1):
    """返回文本数据集的词元索引列表和词表"""
    lines = read_txt(path)
    tokens = tokenize(lines, token=token)
    vocab = Vocab(tokens)
    # 将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def process_agnews(frac=10, max_length=64, min_freq=5):
    """处理AG News数据集并保存子集

    Args:
        frac: 子集比例分母 (1/frac)
        max_length: 文本最大长度（词数）
        min_freq: 最小词频，低于此频率的词将被归为<unk>
    """
    # 设置随机种子保证可重复性
    random.seed(42)

    # 加载原始数据
    df_train = pd.read_csv('../dataset/AG news/train.csv')
    df_val = pd.read_csv('../dataset/AG news/test.csv')

    def process_dataframe(df, frac):
        """处理单个DataFrame：预处理文本+截断文本+均匀抽样"""
        # 先进行抽样，保持原始分布
        class_indices = defaultdict(list)
        for idx, label in enumerate(df['Class Index']):
            class_indices[label].append(idx)

        subset_indices = []
        samples_per_class = len(df) // (len(class_indices) * frac)

        for label in class_indices:
            subset_indices.extend(
                random.sample(class_indices[label], samples_per_class)
            )

        df = df.iloc[subset_indices].copy()

        # 预处理文本：合并标题和描述，去标点，转小写
        df['Text'] = df.apply(
            lambda x: ' '.join(
                (x['Title'] + ' ' + x['Description'])
                .lower()  # 转为小写
                .translate(str.maketrans('', '', string.punctuation))  # 去除标点
                .split()[:max_length]),  # 截断
            axis=1
        )

        # 标签减1（原始1-4 → 转换为0-3）
        df['Class Index'] = df['Class Index'].astype(int) - 1

        return df[['Class Index', 'Text']]

    train_subset = process_dataframe(df_train, frac)
    val_subset = process_dataframe(df_val, frac)

    # 第一步：统计词频
    word_freq = defaultdict(int)
    for text in train_subset['Text'].values:
        for word in text.split():
            word_freq[word] += 1

    # 第二步：构建词汇表，过滤低频词
    vocab = {'<pad>': 0, '<unk>': 1}
    current_idx = 2  # 从2开始，因为0和1已被占用

    # 只保留频率≥min_freq的词，并按频率从大到小排序
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    for word, freq in sorted_words:
        if freq >= min_freq:
            vocab[word] = current_idx
            current_idx += 1

    # 第三步：将文本转换为索引序列，低频词转为<unk>
    def text_to_indices(text):
        indices = [vocab.get(word, vocab['<unk>']) for word in text.split()]
        # 如果索引序列小于max_length，填充0直到长度为max_length
        return indices + [0] * (max_length - len(indices)) if len(indices) < max_length else indices[:max_length]

    train_subset['Indices'] = train_subset['Text'].apply(text_to_indices)
    val_subset['Indices'] = val_subset['Text'].apply(text_to_indices)

    # 保存词汇表
    with open(f"../dataset/AG news/frac{frac}_ml{max_length}_minfreq{min_freq}_vocab.txt", 'w') as f:
        for word, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{word}\t{idx}\t{word_freq.get(word, 0)}\n")

    # 保存训练集和验证集（包含原始文本和索引序列）
    train_subset.to_csv(f"../dataset/AG news/frac{frac}_ml{max_length}_minfreq{min_freq}_train_processed.csv",
                        index=False)
    val_subset.to_csv(f"../dataset/AG news/frac{frac}_ml{max_length}_minfreq{min_freq}_val_processed.csv", index=False)

    # 统计信息
    num_unk = sum(1 for word in word_freq if word_freq[word] < min_freq)
    print(f"处理完成！子集已保存到 ../dataset/AG news/下")
    print(f"词汇表大小: {len(vocab)} (包含{num_unk}个低频词被归为<unk>)")
    print(f"训练子集: {len(train_subset)} 条 (标签范围: {set(train_subset['Class Index'])})")
    print(f"测试子集: {len(val_subset)} 条 (标签范围: {set(val_subset['Class Index'])})")


if __name__ == '__main__':
    import pandas as pd
    import torch
    from collections import defaultdict
    import random
    import string

    process_agnews(frac=2, max_length=64, min_freq=10)

    # # 加载EMNIST训练集
    # full_train = datasets.EMNIST(
    #     root='../dataset',
    #     split='balanced',
    #     train=True,
    #     download=True,
    #     transform=None  # 暂时禁用transform以检查原始数据
    # )
    # full_test = datasets.EMNIST(root='../dataset', split='balanced', train=False, download=True, transform=None)
    #
    # print("类别名称:", full_train.classes)
    # print("类别数量:", len(full_train.classes))
    #
    # print(len(full_train), len(full_test))



