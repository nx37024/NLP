# -*- coding: utf-8 -*-
# @Author:Neo
from keras.src.legacy.preprocessing.text import Tokenizer
import joblib

def dm_one_hot():
    vocabs = {"计算机","数学","科学","人工智能","大模型训练"}
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(vocabs)
    print(tokenizer.word_index)
    print(tokenizer.index_word)

    for vocab in vocabs:
        zero_list = [0] * len(vocabs)
        idx = tokenizer.word_index[vocab] - 1
        zero_list[idx] = 1
        print(vocab, 'one_bot编码：',zero_list)

    #todo:保存训练结果
    mypath = "./mytokenizer"
    joblib.dump(tokenizer,mypath)
    print("保存成功")

def dm_one_hot_use():
    mypath = "./mytokenizer"
    mytokenizer = joblib.load(mypath)
    print(mytokenizer.word_index)


if __name__ == '__main__':
    dm_one_hot()
    dm_one_hot_use()
