import os

import jieba

current_path = os.path.dirname(__file__)

jieba.load_userdict(os.path.join(current_path, "dict.txt"))

# 加载词频
with open(os.path.join(current_path, "dict.txt"), "r", encoding="utf-8") as f:
    for line in f.readlines():
        line = line.strip()
        jieba.suggest_freq(line, tune=True)

if __name__ == '__main__':
    string = "台中正确应该不会被切开"

    res = jieba.cut(string, HMM=False)
    print(list(res))
