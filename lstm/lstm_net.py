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
        self._train_x = []
        self._train_y = []
        self._test_x = []
        self._test_y = []
        self._model_path = "model/"
        self.nb_classes = 0
        self._vocab_size = 0

    def __str__(self):
        return "This is my Lstm model!"

    def build_net(self, **kwargs):
        self._vocab_size = self.init_weight[0].shape[0]
        if self._vocab_size == 0:
            raise Exception('The vocab_size is not be zero!')
        self.model.add(Embedding(
            self._vocab_size,
            kwargs['word_dim'],
            input_length=kwargs['max_len'],
            trainable=False,
            weights=self.init_weight)
        )
        # 使用了堆叠的LSTM架构
        self.model.add(LSTM(units=kwargs['hidden_units'], return_sequences=True))
        self.model.add(LSTM(units=kwargs['hidden_units'], return_sequences=False))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.nb_classes, activation='softmax'))
        self.model.compile(loss=kwargs['loss'], optimizer=kwargs['optimizer'])

    def split_set(self, train_data, train_label):
        """

        :param train_data: 训练词序
        :param train_label: 训练标签
        :return:
        """
        self._train_x, self._test_x, train_y, test_y \
            = train_test_split(np.array(train_data), train_label, test_size=0.1, random_state=1)
        if self.nb_classes == 0:
            raise Exception("The nb_classes is not be zero!")
        self._train_y = np_utils.to_categorical(train_y, self.nb_classes)
        self._test_y = np_utils.to_categorical(test_y, self.nb_classes)
        print("train_x shape: ", self._train_x.shape)
        print("train_y shape: ", self._train_y.shape)
        print("test_x shape: ", self._test_x.shape)
        print("test_y shape: ", self._test_y.shape)

    def model_fit(self, model_file, **kwargs):
        """

        :param model_file: 模型文件名
        :param kwargs: batch_size, epochs
        :return:
        """
        batch_size = kwargs['batch_size']
        epochs = kwargs['epochs']
        self.model.fit(
            self._train_x, self._train_y,
            batch_size=batch_size, epochs=epochs,
            validation_data=(self._test_x, self._test_y)
        )
        try:
            self.model.save(self._model_path + model_file + ".h5")
        except Exception as e:
            print("[model_save]" + str(e))

    def model_fit_on_batch(self, x_train, y_train):
        try:
            self.model.train_on_batch(x=x_train, y=y_train)
        except Exception as e:
            print('[model_train_on_batch]' + str(e))

    def model_load(self, model_file):
        """

        :param model_file: 模型文件名
        :return:
        """
        try:
            self.model = load_model(self._model_path + model_file + ".h5")
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
