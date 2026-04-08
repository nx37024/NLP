# -*- coding: utf-8 -*-
# @Author:Neo
import jieba

#精确分词
def dm1(content):
    str = jieba.lcut(content, cut_all=False)
    return str

#全模式分词
def dm2(content):
    str = jieba.lcut(content, cut_all=True)
    return str

#搜索引擎分词
def dm3(content):
    str = jieba.lcut_for_search(content)
    return str

def dm4(content):
    jieba.load_userdict("./file/userdict.txt")
    str = jieba.lcut_for_search(content)
    return str


if __name__ == '__main__':
    content = "科学学习，NLP自然语言处理中的分词是学习大模型的重点和核心问题，AI学习的最基础要求。"
    dm = dm1(content)
    for v in dm:
        print(v)

    print(f'结果是==》{dm}')

    dm2 = dm2(content)
    print(f'结果是==》{dm2}')

    dm3 = dm3(content)
    print(f'结果是==》{dm3}')

    dm4 = dm4(content)
    print(f'结果是==》{dm4}')


