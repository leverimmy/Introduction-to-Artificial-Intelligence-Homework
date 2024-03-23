import json
import math
import sys
import time
from collections import defaultdict

pinyin2word = {}
word_count = [{} for _ in range(4)]

my_k = 1
my_lambda = 0.95
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
            pinyin_counts_list[key] += word_count[1].get(item, 0)
    return pinyin_counts_list


def get_probability(char_list):
    length = len(char_list)
    prefix = ''.join(char_list[0:-1])
    word = ''.join(char_list)

    if word in word_count[length] and prefix in word_count[length - 1]:
        current_probability = word_count[length][word] / word_count[length - 1][prefix]
    else:
        current_probability = 0
    return my_lambda * current_probability + \
        (1 - my_lambda) * word_count[1].get(char_list[-1], 0) / total_count


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
        f[0][char1] = get_weight(word_count[1].get(char1, 0) / uni_pinyin_count[pinyins[0]])

    # 将第二个拼音对应的状态设为二元模型得到的状态
    for char2 in pinyin2word[pinyins[1]]:
        for char1 in pinyin2word[pinyins[0]]:
            if f[0][char1] + get_weight(get_probability([char1, char2])) < f[1][char2] \
                    or pred[1][char2] is None:
                f[1][char2] = f[0][char1] + get_weight(get_probability([char1, char2]))
                pred[1][char2] = char1

    # 使用二元模型
    if opt == 2:
        for i in range(2, levels):
            pinyin1 = pinyins[i - 1]
            pinyin2 = pinyins[i]
            for char2 in pinyin2word[pinyin2]:
                for char1 in pinyin2word[pinyin1]:
                    if f[i - 1][char1] + get_weight(get_probability([char1, char2])) < f[i][char2] \
                            or pred[i][char2] is None:
                        f[i][char2] = f[i - 1][char1] + get_weight(get_probability([char1, char2]))
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
                        if f[i - 1][char2] + get_weight(get_probability([char1, char2, char3])) < f[i][char3] \
                                or pred[i][char3] is None:
                            f[i][char3] = f[i - 1][char2] + get_weight(get_probability([char1, char2, char3]))
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

    word_count[1] = load_list_from_json('../data/sina_news_gbk/uni_word_count.json')
    word_count[2] = load_list_from_json('../data/sina_news_gbk/bi_word_count.json')
    # word_count[3] = load_list_from_json('../data/sina_news_gbk/tri_word_count.json')

    keys_to_delete = []
    for idx in range(1, 4):
        for key_, value_ in word_count[idx].items():
            if value_ <= my_k:
                keys_to_delete.append((idx, key_))  # 将待删除的键添加到列表中
    # 在迭代结束后删除待删除的键
    for idx, key_ in keys_to_delete:
        del word_count[idx][key_]

    uni_pinyin_count = get_pinyin_count(pinyin2word)
    total_count = sum(uni_pinyin_count.values())

    # 测量总响应时长
    total_response_time = 0

    for input_line in sys.stdin:  # 输入 'qing hua da xue'
        input_line = input_line.strip()
        if not input_line:  # 如果读取到空行，跳过
            break
        # 测量单次响应时长
        start_time = time.time()

        # 使用二元模型或三元模型
        solve(input_line.split(), 2)
        # solve(input_line.split(), 3)
        end_time = time.time()

        total_response_time += end_time - start_time

    sys.stderr.write(str(total_response_time))
