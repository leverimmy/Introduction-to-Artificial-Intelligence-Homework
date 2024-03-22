import json
import math
import sys
from collections import defaultdict

pinyin2word = {}
uni_word_count = {}
bi_word_count = {}
tri_word_count = {}

my_lambda = 0.9
total_number = 0


def load_list_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        element_list = json.load(file)
    return element_list


def get_pinyin_count(element_counts_list):
    pinyin_counts_list = defaultdict(int)
    # 遍历每个键值对，计算 counts 之和，并添加到字典中
    for key, value in element_counts_list.items():
        for item in value:
            pinyin_counts_list[key] += uni_word_count.get(item, 0)
    return pinyin_counts_list


def get_probability2(char1, char2):
    bi_word = f'{char1}{char2}'
    if bi_word in bi_word_count and char1 in uni_word_count:
        current_probability = bi_word_count[bi_word] / uni_word_count[char1]
    else:
        current_probability = 0
    return my_lambda * current_probability + (1 - my_lambda) * uni_word_count.get(char2, 0) / total_count


def get_probability3(char1, char2, char3):
    bi_word = f'{char1}{char2}'
    tri_word = f'{char1}{char2}{char3}'
    if tri_word in tri_word_count and bi_word in bi_word_count:
        current_probability = tri_word_count[tri_word] / bi_word_count[bi_word]
    else:
        current_probability = 0
    return my_lambda * current_probability + (1 - my_lambda) * uni_word_count.get(char3, 0) / total_count


def get_weight(x):
    return float('inf') if x == 0 else -math.log(x)


def solve(pinyins, opt):
    f = {}
    pred = {}
    levels = len(pinyins)

    # 初始化各值和前驱
    for i in range(len(pinyins)):
        f[i] = {}
        pred[i] = {}
        for char in pinyin2word[pinyins[i]]:
            f[i][char] = float('inf')
            pred[i][char] = None

    # 将第一个拼音对应的状态设为 -ln(频率)
    for char1 in pinyin2word[pinyins[0]]:
        f[0][char1] = get_weight(uni_word_count.get(char1, 0) / uni_pinyin_count[pinyins[0]])

    # 将第二个拼音对应的状态设为二元模型得到的状态
    for char2 in pinyin2word[pinyins[1]]:
        for char1 in pinyin2word[pinyins[0]]:
            if f[0][char1] + get_weight(get_probability2(char1, char2)) < f[1][char2]:
                f[1][char2] = f[0][char1] + get_weight(get_probability2(char1, char2))
                pred[1][char2] = char1

    # 使用二元模型
    if opt == 2:
        for i in range(2, levels):
            pinyin1 = pinyins[i - 1]
            pinyin2 = pinyins[i]
            for char2 in pinyin2word[pinyin2]:
                for char1 in pinyin2word[pinyin1]:
                    if f[i - 1][char1] + get_weight(get_probability2(char1, char2)) < f[i][char2]:
                        f[i][char2] = f[i - 1][char1] + get_weight(get_probability2(char1, char2))
                        pred[i][char2] = char1

    # 使用三元模型
    if opt == 3:
        for i in range(2, levels):
            pinyin1 = pinyins[i - 2]
            pinyin2 = pinyins[i - 1]
            pinyin3 = pinyins[i]
            for char3 in pinyin2word[pinyin3]:
                for char2 in pinyin2word[pinyin2]:
                    for char1 in pinyin2word[pinyin1]:
                        if f[i - 1][char2] + get_weight(get_probability3(char1, char2, char3)) < f[i][char3]:
                            f[i][char3] = f[i - 1][char2] + get_weight(get_probability3(char1, char2, char3))
                            pred[i][char3] = char2

    # 从最后一个拼音开始回溯
    final_word = None
    for char in pinyin2word[pinyins[levels - 1]]:
        if final_word is None or f[levels - 1][char] < f[levels - 1][final_word]:
            final_word = char
    elem = final_word
    sentence = []
    for i in range(levels - 1, -1, -1):
        sentence.append(elem)
        elem = pred[i][elem]

    # 返回由列表反转顺序后连接而成的答案
    print(''.join(reversed(sentence)))


if __name__ == '__main__':
    pinyin2word = load_list_from_json('../data/lexicon/pinyin2word.json')
    uni_word_count = load_list_from_json('../data/sina_news_gbk/uni_word_count.json')
    bi_word_count = load_list_from_json('../data/sina_news_gbk/bi_word_count.json')
    tri_word_count = load_list_from_json('../data/sina_news_gbk/tri_word_count.json')
    # print('Finished loading...')

    uni_pinyin_count = get_pinyin_count(pinyin2word)
    total_count = sum(uni_pinyin_count.values())

    for input_line in sys.stdin:  # 输入 'qing hua da xue'
        input_line = input_line.strip()
        if not input_line:  # 如果读取到空行，跳过
            break
        # solve(input_line.split(), 2)
        solve(input_line.split(), 3)
