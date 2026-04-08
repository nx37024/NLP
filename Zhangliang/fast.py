# -*- coding: utf-8 -*-
# @Author: Neo
# @Time: 2024/8/10 19:00
import fasttext

# 使用fasttext的train_unsupervised(无监督训练方法)进行词向量的训练
def train(data_path, model_path):
    model = fasttext.train_unsupervised(data_path)
    model.save_model(model_path)
    # 有效训练词汇量为124M, 共218316个单词
    # Read 124M words
    # Number of words:  218316
    # Number of labels: 0
    # Progress: 100.0% words/sec/thread:   53996 lr:  0.000000 loss:  0.734999 ETA:   0h 0m

def load_data(data_path):
    model = fasttext.load_model(data_path)
    word_vectors = model.get_word_vector("the")
    print(word_vectors)

# 模型超参数设定
def train2(data_path, model_path_hyperparameter):
    model = fasttext.train_unsupervised(data_path, "cbow", dim=300, epoch=1, lr=0.01, thread=8)
    model.save_model(model_path_hyperparameter)

if __name__ == '__main__':
    # data_path = r"data/enwik9_dispose"
    # model_path = r"data/enwik9_dispose.bin"
    # #train(data_path, model_path)
    #
    # load_data(model_path)
    data_path = r"data/enwik9_dispose"
    model_path_hyperparameter = r"data/enwik9_dispose_hyperparameter.bin"
    train2(data_path, model_path_hyperparameter)
