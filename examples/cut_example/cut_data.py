import os
import re

import jieba

current_path = os.path.dirname(__file__)
# 字典路径
dict_path = os.path.join(current_path, "dict.txt")

jieba.load_userdict(dict_path)


def merge_two_list(l1, l2):
    """合并列个列表"""
    res = []
    l1_len = len(l1)
    l2_len = len(l2)
    min_len = min(l1_len, l2_len)
    for i in range(min_len):
        res.append(l1[i])
        res.append(l2[i])

    if l1_len > l2_len:
        for i in range(min_len, l1_len):
            res.append(l1[i])
    else:
        for i in range(min_len, l2_len):
            res.append(l2[i])

    return res


if __name__ == '__main__':
    text_path = os.path.join(current_path, "text.txt")
    # 匹配罗马字符
    regex_1 = re.compile(r"(?:[^\u4e00-\u9fa5（）*&……%￥$，,。.@! ！]){1,5}期")
    # 匹配百分比
    regex_2 = re.compile(r"(?:[0-9]{1,3}[.]?[0-9]{1,3})%")

    # 分词之后的结果
    cut_res = []
    # 读取文本
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            result_1 = regex_1.findall(line)
            if result_1:
                line = regex_1.sub("FLAG1", line)

            result_2 = regex_2.findall(line)
            if result_2:
                line = regex_2.sub("FLAG2", line)

            words = jieba.cut(line)
            result = " ".join(words)
            if "FLAG1" in result:
                result = result.split("FLAG1")
                result = merge_two_list(result, result_1)
                result = " ".join(result)
            if "FLAG2" in result:
                result = result.split("FLAG2")
                result = merge_two_list(result, result_2)
                result = " ".join(result)

            cut_res.append(result)

    # 写入结果
    res_path = os.path.join(current_path, "cut_result.txt")
    with open(res_path, "w", encoding="utf-8") as f:
        for item in cut_res:
            f.write(item)
