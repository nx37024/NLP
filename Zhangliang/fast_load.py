# -*- coding: utf-8 -*-
# @Author:https://blog.csdn.net/qq_60735796/article/details/141100435
import fasttext

def load(model_path):
    # 可以使用以下代码加载已经训练好的模型
    print("start the model effect test")
    print("*" * 120)
    model = fasttext.load_model(model_path)
    sports = model.get_nearest_neighbors('sports')
    music = model.get_nearest_neighbors('music')
    dog = model.get_nearest_neighbors('dog')
    affection = model.get_nearest_neighbors('affection')
    print("sports series: ", sports)
    print("-" * 120)
    print("music series: ", music)
    print("-" * 120)
    print("dog series: ", dog)
    print("-" * 120)
    print("affection series: ", affection)

if __name__ == '__main__':
    model_path = r"data/enwik9_dispose.bin"
    model_path_hyperparameter = r"data/enwik9_dispose_hyperparameter.bin"
    load(model_path_hyperparameter)
