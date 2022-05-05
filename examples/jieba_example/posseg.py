from jieba import posseg

if __name__ == '__main__':
    string = "是广泛使用的中文分词工具，具有以下特点"

    for word, flag in posseg.cut(string):
        print(f"{word} {flag}")
