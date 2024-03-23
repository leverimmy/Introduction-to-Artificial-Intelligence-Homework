# README

本文件介绍程序的运行方式和文件结构。

## 文件结构

```
Pinyin/Hand-in/
│  README.md
│
├─data
│  │  input.txt
│  │  output.txt
│  │
│  ├─lexicon
│  │      1st_2nd_order_characters.txt
│  │      pinyin2word.json
│  │      pinyin2word_all.txt
│  │
│  ├─sina_news_gbk
│  │      2016-04.txt
│  │      2016-05.txt
│  │      2016-06.txt
│  │      2016-07.txt
│  │      2016-08.txt
│  │      2016-09.txt
│  │      2016-10.txt
│  │      2016-11.txt
│  │      bi_word_count.json
│  │      README.txt
│  │      tri_word_count.json
│  │      uni_word_count.json
│  │
│  └─std_data
│          std_input.txt
│          std_output.txt
│
└─src
        pinyin.py
        preprocess.py
        test.py
```

其中 `Pinyin/Hand-in/data` 文件夹可以从[清华云盘下载链接](https://cloud.tsinghua.edu.cn/d/876ab9066cbd4f53b168/)或 [GitHub Release](https://github.com/LeverImmy/Introduction-to-Artificial-Intelligence/releases/download/Pinyin/data.zip) 中下载得到，或是通过将下发文件按照上图结构进行放置而得到：图中的 `1st_2nd_order_characters.txt` 即为下发文件中的 `一二级汉字表.txt`，`pinyin2word_all.txt` 即为下发文件中的 `拼音汉字表.txt`。

## 运行方式

假设当前处于 `Pinyin/Hand-in/src` 文件夹下：

- 对数据进行预处理：

  ```bash
  py ./preprocess.py
  ```

  该命令将重新生成 `uni_word_count.json`、`bi_word_count.json` 和 `tri_word_count.json`，预计完成时间为 $10.5$ 分钟。

- 对测试集进行测试：

  ```bash
  py ./pinyin.py < ../data/std_data/std_input.txt > test_output.txt
  ```

  该命令将会在 `Pinyin/Hand-in/src/` 下生成 `test_output.txt`，为测试集对应输出。

  也可以直接使用 `test.py` 进行测试：

  ```bash
  py ./test.py
  ```

  该命令将会以对比的形式，展示输出与标准答案不同的部分，同时统计句正确率、字正确率、平均单次响应时长和总响应时长。

