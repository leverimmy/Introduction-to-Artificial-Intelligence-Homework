import os

if __name__ == '__main__':

    # 比较模型
    os.system('python main.py -c CNN')
    os.system('python main.py -c RNN_LSTM')
    os.system('python main.py -c RNN_GRU')
    os.system('python main.py -c MLP')
    os.system('python main.py -c BerT')

    # 比较参数：learning rate
    rates = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    for rate in rates:
        os.system(f'python main.py -l {rate}')

    # 比较参数：batch size
    sizes = [20, 35, 50, 65, 80]
    for size in sizes:
        os.system(f'python main.py -b {size}')