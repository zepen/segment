# -*- coding: utf-8 -*-
from pickle import load
from lstm.lstm_net import LongShortTMNet, np
from lstm.processing import DataProcessing

# 加载入初始化向量
with open('model/init_weight_wv.pickle', 'rb') as f:
    init_weight_wv = load(f)
# 加载词典
with open('dictionary/word2idx.pickle', 'rb') as f:
    word2idx = load(f)


def train_lstm_run():
    data_processing = DataProcessing()
    data_processing.load_file('corpus/msr.txt')
    data_processing.train_transform(word2idx)
    # 获取训练数据
    train_word_num = data_processing.get_train_word_num()
    train_label = data_processing.get_train_label()
    nb_classes = len(np.unique(train_label))
    # stacking LSTM
    model_name = 'lstm_model.h5'
    net = LongShortTMNet()
    net.init_weight = [np.array(init_weight_wv)]
    net.nb_classes = nb_classes
    net.split_set(train_word_num, train_label)
    print("Train...")
    net.build_net()
    net.model_fit(model_name, batch_size=128, epochs=20)

if __name__ == '__main__':
    # train lstm
    train_lstm_run()
