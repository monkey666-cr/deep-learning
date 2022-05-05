"""
词性标注代码实现及信息提取
"""
import os

from jieba import posseg

current_path = os.path.dirname(__file__)

if __name__ == '__main__':
    with open(os.path.join(current_path, "extract_output.txt"), "w", encoding="utf-8") as fo:
        with open(os.path.join(current_path, "text2.txt"), "r", encoding="utf-8") as fp:
            for line in fp.readlines():
                line = line.strip()
                if len(line) > 0:
                    fo.write(
                        " ".join([f"{word} {flag}" for word, flag in posseg.cut(line)])
                    )
                    fo.write("\n")
