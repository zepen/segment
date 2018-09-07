# -*- coding:utf-8 -*-
"""
定义模型类
"""
import os
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import TensorBoard, EarlyStopping
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
        self._graph = None

    def __str__(self):
        return "This is my Lstm model!"

    def build_net(self, **kwargs):
        """  建立模型

        :param kwargs:
        :return:
        """
        self._vocab_size = self.init_weight[0].shape[0]
        if self._vocab_size == 0:
            raise Exception('The vocab_size is not be zero!')
        self.model.add(Embedding(
            self._vocab_size,
            kwargs['word_dim'],
            input_length=kwargs['max_len'],
            trainable=False,
            weights=self.init_weight,
            name='embed_layer'
           )
        )
        # 使用了堆叠的LSTM架构
        self.model.add(LSTM(units=kwargs['hidden_units'], return_sequences=True, name='lstm_layer_1'))
        self.model.add(LSTM(units=kwargs['hidden_units'], return_sequences=False, name='lstm_layer_2'))
        add_layers = set(kwargs.keys()) & set(['add_layer_' + str(i) for i in range(100)])
        if len(add_layers) != 0:
            add_layers = list(add_layers)
            for layer in add_layers:
                try:
                    self._add_layer(layer)
                except Exception as e:
                    print("[INFO]" + str(e))
                    break
        else:
            print("** Use default model **")
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.nb_classes, activation='softmax', name='output_layer'))
        self.model.compile(
            loss=kwargs['loss'],
            optimizer=kwargs['optimizer'],
            metrics=['accuracy']
        )

    def _add_layer(self, layer):
        """

        :param layer: 添加新层
        :return:
        """
        if isinstance(layer, Dense):
            self.model.add(layer)
        raise Exception("The layer object is not be suppose!")

    def get_model_structure(self):
        """ 获取模型结构
        :return:
        """
        self.model.summary()

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
        for s in zip(['train_x_shape', 'train_y_shape', 'test_x_shape', 'test_y_shape'],
                     [self._train_x, self._train_y, self._test_x, self._test_x]):
            print(s[0], ":", s[1].shape)

    def model_fit(self, model_file, **kwargs):
        """

        :param model_file: 模型文件名
        :param kwargs: batch_size, epochs
        :return:
        """
        batch_size = kwargs['batch_size']
        epochs = kwargs['epochs']
        if os.path.exists('logs/') is False:
            os.mkdir('logs/')
        tensor_board = TensorBoard(embeddings_layer_names='embed_layer')
        early_stop = EarlyStopping(patience=10)
        self.model.fit(
            self._train_x, self._train_y,
            batch_size=batch_size, epochs=epochs,
            validation_data=(self._test_x, self._test_y),
            callbacks=[tensor_board, early_stop]
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
            self._graph = tf.get_default_graph()
            self.model = load_model(self._model_path + model_file + ".h5", compile=False)
            print("[INFO] The Lstm model is load!")
        except Exception as e:
            print("[model_load]" + str(e))

    def predict_label(self, input_num, label_dict):
        """ 预测标签处理，将常理不能发生的标注概率重置为零

        :param input_num:
        :param label_dict:
        :return:
        """
        input_num = np.array(input_num)

        predict_prob, predict_label = None, None
        try:
            with self._graph.as_default():
                predict_prob = self.model.predict_proba(input_num)
        except Exception as e:
            print('[predict_prob]' + str(e))
        try:
            with self._graph.as_default():
                predict_label = self.model.predict_classes(input_num)
                print("predict_label:%s" % predict_label)
        except Exception as e:
            print('[predict_label]' + str(e))
        if (predict_label is not None) or (predict_prob is not None):
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
                predict_label[i+1] = predict_prob[i+1].argmax()
            return predict_label

    def cut_word(self, input_num, input_txt, label_dict, num_dict):
        """

        :param input_num: 输入词序列
        :param input_txt: 输入文本
        :param label_dict: 输入{文本标签:数值标签}
        :param num_dict: 输入{数值标签:文本标签}
        :return:
        """
        predict_label = self.predict_label(input_num, label_dict=label_dict)
        print("predict_label:%s" % predict_label)
        if predict_label is not None:
            predict_label_new = [num_dict[x] for x in predict_label]
            predict_str = []
            zip_ = [x for x in zip(input_txt, predict_label_new)]
            for index, v in enumerate(zip_):
                w, l = v
                if l == 'S':
                    predict_str.append(w)
                elif l == 'B':
                    str_ = ''
                    while index < len(zip_):
                        str_ += zip_[index][0]
                        index += 1
                        if index == len(zip_):
                            break
                        if zip_[index][1] == 'E':
                            str_ += zip_[index][0]
                            break
                    predict_str.append(str_)
            print("predict_str:%s" % predict_str)
            return " | ".join(predict_str)
        else:
            return ""
