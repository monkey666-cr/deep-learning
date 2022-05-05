import os

from stanfordcorenlp import StanfordCoreNLP

current_dir = os.path.dirname(__file__)
models_path = os.path.join(current_dir, "stanford-corenlp-4.4.0")

news_path = os.path.join(current_dir, "news.txt")


def write_res(file_name, res):
    file_path = os.path.join(current_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in res:
            f.write(" ".join(item))
            f.write("\n")


def main():
    ner_res = []
    tag_res = []

    with StanfordCoreNLP(models_path, lang="zh") as nlp:
        with open(news_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) < 1:
                    continue
                ner_res.append(
                    [each[0] + "/" + each[1] for each in nlp.ner(line) if len(each) == 2]
                )
                tag_res.append(
                    each[0] + "/" + each[1] for each in nlp.pos_tag(line) if len(each) == 2
                )

    write_res("ner.txt", ner_res)
    write_res("pos_tag.txt", tag_res)


if __name__ == '__main__':
    main()
