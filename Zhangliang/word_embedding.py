import torch
from keras.src.legacy.preprocessing.text import Tokenizer
from torch.utils.tensorboard import SummaryWriter
import jieba
import torch.nn as nn
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

def dm_one():
    sent1 = '文本张量表示方法，是学习大模型的必修课，训练模型需要非常好的词汇文本'
    sent2 = 'Neo正在学自然语言处理，学习语言模型需要一些捷径方法，快速上手实践非常重要'
    sents = [sent1, sent2]

    # 分词
    word_list = []
    for sent in sents:
        word_list.append(jieba.lcut(sent))

    # 标下标（对句子文本数值化）
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(word_list)

    # 提取文本
    tokens = tokenizer.index_word.values()

    # 创建nn.Embedding层
    embd = nn.Embedding(num_embeddings=len(tokens), embedding_dim=8)
    print(embd.weight)

    summarywriter = SummaryWriter()
    summarywriter.add_embedding(embd.weight.data, tokens)
    summarywriter.close()

    #启动服务
    #tensorboard --logdir=runs --host 0.0.0.0 --port=6006

    # for idx in range(len(tokenizer.index_word)):
    #     tmpvec = embd(torch.tensor(idx))
    #     print('%4s' % (tokenizer.index_word[idx+1]),tmpvec.detach().numpy())


if __name__ == '__main__':
    dm_one()
