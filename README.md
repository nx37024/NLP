# 大模型-自然语言处理
## 第01课
### Pycharm工具地址：
https://download-cdn.jetbrains.com/python/pycharm-2026.1.exe

### Conda安装地址：
https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_24.1.2-0-Windows-x86_64.exe



### 训练模型代码示例：
```
def dm_one_hot():
    vocabs = {"计算机","数学","科学","人工智能"}
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(vocabs)
    print(tokenizer.word_index)
    print(tokenizer.index_word)

    for vocab in vocabs:
        zero_list = [0] * len(vocabs)
        idx = tokenizer.word_index[vocab] - 1
        zero_list[idx] = 1
        print(vocab, 'one_bot编码是：',zero_list)

    #todo:保存训练结果
    mypath = "./mytokenizer"
    joblib.dump(tokenizer,mypath)
    print("保存成功")
```