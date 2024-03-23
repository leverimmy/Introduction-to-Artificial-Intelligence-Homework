import json
import re
from collections import defaultdict
from tqdm import tqdm


def get_pinyin2word():
    pinyin2word_mapping = {}
    with open('../data/lexicon/pinyin2word_all.txt', 'r', encoding='gbk') as file:
        for line in file:
            line_segments = line.strip().split()
            pinyin2word_mapping[line_segments[0]] = line_segments[1:]
    with open("../data/lexicon/pinyin2word.json", "w", encoding="utf-8") as json_file:
        json.dump(pinyin2word_mapping, json_file, ensure_ascii=False, indent=4)


def get_word_count():
    uni_word_counter = defaultdict(int)
    bi_word_counter = defaultdict(int)
    tri_word_counter = defaultdict(int)

    # 枚举每个文件的文件名
    for i in tqdm(range(4, 12)):
        with open(f'../data/sina_news_gbk/2016-{str(i).zfill(2)}.txt', 'r', encoding='gbk') as file:
            for line in tqdm(file):
                try:
                    data = json.loads(line).get('html', "")

                    delimiter_pattern = re.compile(r'[^\u4e00-\u9fa5]+')
                    segments = delimiter_pattern.split(data)

                    for segment in segments:
                        # 去除空字符串
                        segment = segment.strip()
                        if not segment:
                            continue

                        # 截取长度为 1 的字符串
                        for j in range(len(segment)):
                            uni_word_counter[segment[j:j + 1]] += 1
                        # 截取长度为 2 的字符串
                        for j in range(len(segment) - 1):
                            bi_word_counter[segment[j:j + 2]] += 1
                        # 截取长度为 3 的字符串
                        for j in range(len(segment) - 2):
                            tri_word_counter[segment[j:j + 3]] += 1
                except:
                    pass

    # 将字典写入到 JSON 文件中
    with open("../data/sina_news_gbk/uni_word_count.json", "w", encoding="utf-8") as json_file:
        json.dump(uni_word_counter, json_file, ensure_ascii=False, indent=4)
    with open("../data/sina_news_gbk/bi_word_count.json", "w", encoding="utf-8") as json_file:
        json.dump(bi_word_counter, json_file, ensure_ascii=False, indent=4)
    with open("../data/sina_news_gbk/tri_word_count.json", "w", encoding="utf-8") as json_file:
        json.dump(tri_word_counter, json_file, ensure_ascii=False, indent=4)


get_pinyin2word()  # 获取 pinyin2word.json
get_word_count()  # 获取 uni_word_count.json, bi_word_count.json, tri_word_count.json
