import subprocess

result = subprocess.run("py ./pinyin.py < ../data/std_data/std_input.txt > test_output.txt",
                        shell=True, capture_output=True, text=True)

my_file = []
std_file = []

with open('./test_output.txt', 'r', encoding='gbk') as file:
    for line in file:
        my_file.append(line.strip())

with open('../data/std_data/std_output.txt', 'r', encoding='utf-8') as file:
    for line in file:
        std_file.append(line.strip())

total_sentence_count = 0
accepted_sentence_count = 0
total_character_count = 0
accepted_character_count = 0

for my_sentence, std_sentence in zip(my_file, std_file):
    total_sentence_count += 1
    total_character_count += len(my_sentence)
    if my_sentence == std_sentence:
        accepted_sentence_count += 1
    else:
        print(f'<{my_sentence} != {std_sentence}>')
    for my_character, std_character in zip(my_sentence, std_sentence):
        if my_character == std_character:
            accepted_character_count += 1

print(f'句正确率：{accepted_sentence_count / total_sentence_count * 100} %')
print(f'字正确率：{accepted_character_count / total_character_count * 100} %')
