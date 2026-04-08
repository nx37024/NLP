# 大模型-自然语言处理
## 常用工具
### Pycharm工具地址：
https://download-cdn.jetbrains.com/python/pycharm-2026.1.exe

### Conda安装地址：
https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_24.1.2-0-Windows-x86_64.exe



### 训练模型 Demo：
```
def dm_one_hot():
    vocabs = {"Natural","Learning","Language","Course"}
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(vocabs)

    for vocab in vocabs:
        zero_list = [0] * len(vocabs)
        idx = tokenizer.word_index[vocab] - 1
        zero_list[idx] = 1
        print(vocab, 'one_bot：',zero_list)

    #todo:save
    mypath = "./mytokenizer"
    joblib.dump(tokenizer,mypath)
    print("done")
```