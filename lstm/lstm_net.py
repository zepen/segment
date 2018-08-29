"""
定义模型类
"""
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from sklearn.model_selection import train_test_split


class LongShortTMNet(object):

    def __init__(self):
        self.model = Sequential()
        self.init_weight = []
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
        self.batch_size = 128
        self.word_dim = 100
        self.max_len = 7
        self.hidden_units = 100
        self.nb_classes = 0
        self.model_path = "model/"

    def __str__(self):
        return "This is my Lstm model!"

    def build_net(self):
        print('stacking LSTM...')
        self.model.add(Embedding(
            self.init_weight[0].shape[0],
            self.word_dim,
            input_length=self.max_len,
            trainable=False,
            weights=self.init_weight)
        )
        # 使用了堆叠的LSTM架构
        self.model.add(LSTM(units=self.hidden_units, return_sequences=True))
        self.model.add(LSTM(units=self.hidden_units, return_sequences=False))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.nb_classes, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    def split_set(self, train_word_num, train_label):
        """

        :param train_word_num: 训练词序
        :param train_label: 训练标签
        :return:
        """
        self.train_x, self.test_x, train_y, test_y \
            = train_test_split(train_word_num, train_label, train_size=0.9, random_state=1)
        self.train_y = np_utils.to_categorical(train_y, self.nb_classes)
        self.test_y = np_utils.to_categorical(test_y, self.nb_classes)

    def model_fit(self, model_file, **kwargs):
        """

        :param model_file: 模型文件名
        :param kwargs: batch_size, epochs
        :return:
        """
        batch_size = kwargs['batch_size']
        epochs = kwargs['epochs']
        self.model.fit(
            self.train_x, self.train_y,
            batch_size=batch_size, epochs=epochs,
            validation_data=(self.test_x, self.test_y)
        )
        try:
            self.model.save(self.model_path + model_file + ".h5")
        except Exception as e:
            print("[model_save]" + str(e))

    def model_load(self, model_file):
        """

        :param model_file: 模型文件名
        :return:
        """
        try:
            self.model = load_model(self.model_path + model_file + ".h5")
        except Exception as e:
            print("[model_load]" + str(e))

    def predict_label(self, input_num, label_dict):
        """

        :param input_num:
        :param label_dict:
        :return:
        """
        input_num = np.array(input_num)
        predict_prob = self.model.predict_proba(input_num, verbose=False)
        predict_label = self.model.predict_classes(input_num, verbose=False)
        for i, label in enumerate(predict_label[:-1]):
            if i == 0:  # 如果首字， 不可为E， M
                predict_prob[i, label_dict[u'E']] = 0
                predict_prob[i, label_dict[u'M']] = 0
            if label == label_dict[u'B']:  # 前字为B， 后字不可为B， S
                predict_prob[i + 1, label_dict[u'B']] = 0
                predict_prob[i + 1, label_dict[u'S']] = 0
            if label == label_dict[u'E']:  # 前字为E， 后字不可为M， E
                predict_prob[i + 1, label_dict[u'M']] = 0
                predict_prob[i + 1, label_dict[u'E']] = 0
            if label == label_dict[u'M']:  # 前字为M， 后字不可为B， S
                predict_prob[i + 1, label_dict[u'B']] = 0
                predict_prob[i + 1, label_dict[u'S']] = 0
            if label == label_dict[u'S']:  # 前字为S， 后字不可为M， E
                predict_prob[i + 1, label_dict[u'M']] = 0
                predict_prob[i + 1, label_dict[u'E']] = 0
            # 前面处理将常理不能标注的发生的概率重置为零
            predict_label[i+1] = predict_prob[i+1].argmax()
        return predict_label

    def cut_word(self, input_num, input_txt, label_dict, num_dict):
        predict_label = self.predict_label(input_num, label_dict=label_dict)
        predict_label_new = [num_dict[x] for x in predict_label]
        predict_str = []
        zip_ = zip(input_txt, predict_label_new)
        for index, v in enumerate(zip_):
            w, l = v
            if l == 'S':
                predict_str.append(w)
            elif l == 'B':
                str_ = ''
                while 1:
                    str_ += zip_[index][0]
                    index += 1
                    if zip_[index][1] == 'E':
                        str_ += zip_[index][0]
                        break
                predict_str.append(str_)
        return predict_str
