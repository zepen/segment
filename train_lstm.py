# -*- coding: utf-8 -*-
from pickle import load
from lstm.lstm_net import LongShortTMNet, np, os
from lstm.processing import DataProcessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

INIT_WEIGHT_WV_PATH = 'model/init_weight_wv.pickle'
VOCAB_DICT_PATH = 'dictionary/word2idx.pickle'

# 加载入初始化向量
if os.path.exists(INIT_WEIGHT_WV_PATH):
    with open(INIT_WEIGHT_WV_PATH, 'rb') as f:
        init_weight_wv = load(f)
    print("[INFO] The init_weight_wv is load!")
else:
    print("[WARN] The init_weight_wv.pickle is not exist! so must be run train_word2vec.py")
# 加载词典
if os.path.exists(VOCAB_DICT_PATH):
    with open(VOCAB_DICT_PATH, 'rb') as f:
        vocab_dict = load(f)
    print("[INFO] The vocab_dict is load!")
else:
    print("[WARN] The word2idx.pickle is not exist, so must be run train_word2vec.py")


def train_lstm_run(model_name):
    """

    :param model_name: 训练模型文件名
    :return:
    """
    # 转换训练数据
    data_processing = DataProcessing()
    data_processing.load_file('corpus/msr.txt')
    data_processing.train_transform(vocab_dict, 0)
    # 获取训练数据
    train_data = data_processing.get_train_data()
    train_label = data_processing.get_train_label()
    nb_classes = len(np.unique(train_label))
    # 创建并训练模型
    net = LongShortTMNet()
    net.init_weight = [np.array(init_weight_wv)]
    net.nb_classes = nb_classes
    net.split_set(train_data, train_label)
    print("The model train is begin...")
    net.build_net(
        word_dim=100,
        max_len=7,
        hidden_units=512,
        loss='categorical_crossentropy',
        optimizer='adam'
    )
    net.model_fit(
        model_file=model_name,
        batch_size=128,
        epochs=30
    )


if __name__ == '__main__':
    model_file = 'lstm_model'
    # train lstm
    train_lstm_run(model_file)
