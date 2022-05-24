from gensim.models import Word2Vec
import numpy as np
import warnings
warnings.filterwarnings('ignore')  # 警告扰人，手动封存
import torch
import torch.utils.data
import torch.nn.functional as F
import json
import re
import operator
import functools
import time
import math
import torch.nn as nn
from torch.autograd import Variable
from itertools import chain

class Word_Embedding(object):
    def __init__(self, path, vocab):
        """
        :param path: string; the path of word embedding word2vec模型的路径
        :param vocab: object; 上面构建的词典
        """
        print("[INFO] Loading external word embedding...")
        # self.path = path
        self.vocablist = vocab.word_list()
        # 中文key
        self.vocab = vocab
        self.count = 0
        self.model = Word2Vec.load(path)
        self.vocabsum = list(self.model.wv.vocab.keys())
        print("Loading Finish...")

    def load_my_vecs(self):
        """Load word embedding"""
        word_vecs = {}
        for word in self.vocablist:
            if word in self.vocabsum:
                word_vecs[word] = self.model[word]
                self.count = self.count + 1
            else:
                continue
        return word_vecs

    # 采用 0 来填充word2vec模型中不存在的words
    def add_unknown_words_by_zero(self, word_vecs, k):
        """Solve unknown by zeros"""
        zero = [0.0] * k
        list_word2vec = []
        oov = 0
        iov = 0
        for i in range(self.vocab.size()):
            word = self.vocab.id2word(i)
            if word not in word_vecs:
                oov += 1
                word_vecs[word] = zero
                list_word2vec.append(word_vecs[word])
            else:
                iov += 1
                list_word2vec.append(word_vecs[word])
        # logger.info("[INFO] oov count %d, iov count %d", oov, iov)
        return list_word2vec

    # 采用平均向量来填充word2vec模型中不存在的words
    def add_unknown_words_by_avg(self, word_vecs, k):
        """Solve unknown by avg word embedding"""
        # solve unknown words inplaced by zero list
        word_vecs_numpy = []
        for word in self.vocablist:
            if word in word_vecs:
                word_vecs_numpy.append(word_vecs[word])
        col = []
        for i in range(k):
            sum = 0.0
            for j in range(int(len(word_vecs_numpy))):
                sum += word_vecs_numpy[j][i]
                sum = round(sum, 6)
            col.append(sum)
        zero = []
        for m in range(k):
            avg = col[m] / int(len(word_vecs_numpy))
            avg = round(avg, 6)
            zero.append(float(avg))

        list_word2vec = []
        oov = 0
        iov = 0
        for i in range(self.vocab.size()):
            word = self.vocab.id2word(i)
            if word not in word_vecs:
                oov += 1
                word_vecs[word] = zero
                list_word2vec.append(word_vecs[word])
            else:
                iov += 1
                list_word2vec.append(word_vecs[word])
        # logger.info("[INFO] External Word Embedding iov count: %d, oov count: %d", iov, oov)
        return list_word2vec

    # 采用随机数来填充word2vec模型中不存在的words
    def add_unknown_words_by_uniform(self, word_vecs, k):
        """Solve unknown word by uniform(-0.25,0.25)"""
        uniform = 0.25
        list_word2vec = []
        oov = 0
        iov = 0
        for i in range(self.vocab.size()):
            word = self.vocab.id2word(i)
            if word not in word_vecs:
                oov += 1
                word_vecs[word] = np.random.uniform(-1 * uniform, uniform, k).round(6).tolist()
                list_word2vec.append(word_vecs[word])
            else:
                iov += 1
                list_word2vec.append(word_vecs[word])
        # logger.info("[INFO] oov count %d, iov count %d", oov, iov)
        return list_word2vec