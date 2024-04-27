import os

if __name__ == '__main__':

    # 使用os模块执行命令
    os.system('python main.py -c CNN')
    os.system('python main.py -c RNN_LTSM')
    os.system('python main.py -c RNN_GRU')
    os.system('python main.py -c MLP')
    os.system('python main.py -c BerT')
