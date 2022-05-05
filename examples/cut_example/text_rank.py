import os

from jieba import analyse

current_path = os.path.dirname(__file__)

if __name__ == '__main__':
    with open(os.path.join(current_path, "text2.txt"), "r", encoding="utf-8") as f:
        for x, w in analyse.textrank(f.read(), 10, withWeight=True, withFlag=True):
            print(f"{x}, {w}")
