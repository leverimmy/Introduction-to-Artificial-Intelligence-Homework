import json
import math
import sys

my_lambda = 0.95
total_count = 0


def read_word2pinyin(file_path):
    character_mapping = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line_segments = line.strip().split()
            character_mapping[line_segments[0]] = line_segments[1]
    return character_mapping


def read_counts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        counts_list = json.load(file)
    return counts_list


def get_element_count(element_counts_list):
    element_count = {}
    for target_data in element_counts_list.values():
        elements = target_data.get('words', [])
        counts = target_data.get('counts', [])
        for element, count in zip(elements, counts):
            element_count[element] = count
    return element_count


def get_pinyin_count(element_counts_list):
    pinyin_counts_list = {}
    # 遍历每个键值对，计算 counts 之和，并添加到字典中
    for key, value in element_counts_list.items():
        pinyin_counts_list[key] = sum(value["counts"])
    return pinyin_counts_list


def get_probability(char1, char2):
    word = f'{char1} {char2}'
    if word in word_count:
        current_probability = word_count[word] / character_count[char1]
    else:
        current_probability = 0
    return my_lambda * current_probability + (1 - my_lambda) * character_count[char2] / total_count


def solve(pinyins):
    levels = len(pinyins)
    f = {}
    pred = {}

    for i in range(levels):
        f[i] = {}
        pred[i] = {}
        for char2 in character_counts_list.get(pinyins[i], {}).get("words", []):
            f[i][char2] = float('inf')  # 初始状态设为正无穷大
            pred[i][char2] = None

    # 将第一个拼音对应的状态设为 0
    for char1 in character_counts_list.get(pinyins[0], {}).get("words", []):
        f[0][char1] = -math.log(character_count[char1] / pinyin_single_count[pinyins[0]])

    for i in range(1, levels):
        pinyin1 = pinyins[i - 1]
        pinyin2 = pinyins[i]
        for char2 in character_counts_list.get(pinyin2, {}).get("words", []):
            for char1 in character_counts_list.get(pinyin1, {}).get("words", []):
                if f[i - 1][char1] - math.log(get_probability(char1, char2)) < f[i][char2]:
                    f[i][char2] = f[i - 1][char1] - math.log(get_probability(char1, char2))
                    pred[i][char2] = char1

    # 从最后一个拼音开始回溯
    ans = float('inf')
    final = '啊'
    for char in character_counts_list.get(pinyins[levels - 1], {}).get("words", []):
        if f[levels - 1][char] < ans:
            ans = f[levels - 1][char]
            final = char
    elem = final
    sentence = []
    for i in range(levels - 1, -1, -1):
        sentence.append(elem)
        elem = pred[i][elem]

    # print(sentence)
    print(''.join(reversed(sentence)))


if __name__ == '__main__':
    character_map = read_word2pinyin('./word2pinyin.txt')
    character_counts_list = read_counts('./1_word.txt')
    word_counts_list = read_counts('./2_word.txt')

    character_count = get_element_count(character_counts_list)
    word_count = get_element_count(word_counts_list)
    pinyin_single_count = get_pinyin_count(character_counts_list)
    total_count = sum(pinyin_single_count.values())

    # print(pinyin_single_count)
    # print(total_count)

    # print(character_map['暧'])  # 输出 'ai'
    # count1 = character_count['嫩']
    # print(f'字符"{'嫩'}"出现的次数为: {count1}')  # 输出 '4006'
    # count2 = word_count['巴 嫩']
    # print(f'词语"{'巴嫩'}"出现的次数为: {count2}')  # 输出 '548'

    for input_line in sys.stdin:  # 输入 'qing hua da xue'
        input_line = input_line.strip()
        if not input_line:  # 如果读取到空行，跳过
            break
        solve(input_line.split())
