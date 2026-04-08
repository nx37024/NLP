# -*- coding: utf-8 -*-
# @Author:Neo
import jieba.posseg as pseg


if __name__ == '__main__':
    ps = pseg.lcut('我正在学习自然语言技术')
    print(ps)