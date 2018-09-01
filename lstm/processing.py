# -*- coding: utf-8 -*-
"""
定义数据预处理类
"""
import re
import os
import sys
import codecs
import psutil
from pickle import dump


class DataProcessing(object):

    def __init__(self):
        self._tag_corpus_path = 'corpus/msr.tagging.utf8'
        self._train_data_path = 'train_data/'
        self._dictionary_path = 'dictionary/'
        self.input_text = None
        self.input_data = None
        self.output_data = None
        self._x_data = []
        self._train_label = []
        self._train_data = []
        self._pattern_num = re.compile(r'[0-9]+')
        self.key_u = u'U'
        self.key_p = u'P'
        self.status = 0

    def __str__(self):
        return "This is data processing!"

    def load_file(self, input_file):
        """

        :param input_file: 输入预料文件
        :return:
        """
        self.input_data = codecs.open(input_file, 'r', 'utf-8')
        try:
            self.input_text = self.input_data.readlines()
        except Exception as e:
            print("[load_file]" + str(e))

    def _character_tagging(self):
        """ 加入标注标签 B M E S
        :return:
        """
        self.output_data = codecs.open(self._tag_corpus_path, 'w', 'utf-8')
        if self.input_data is not None:
            print("character tagging...")
            for line in self.input_text:
                word_list = line.strip().split()
                for word in word_list:
                    if len(word) == 1:
                        self.output_data.write(word + "/S ")
                    else:
                        self.output_data.write(word[0] + "/B ")
                        for w in word[1:len(word) - 1]:
                            self.output_data.write(w + "/M ")
                        self.output_data.write(word[len(word) - 1] + "/E ")
                self.output_data.write("\n")
            print("*** The tag corpus is completed! ***")
            try:
                self.input_data.close()
            except Exception as e:
                print('[processing_close_input_data]' + str(e))
            try:
                self.output_data.close()
            except Exception as e:
                print('[processing_close_output_data]' + str(e))
        else:
            raise Exception('The input data is None object!')

    def _save_train_data(self, num):
        """

        :param num: 文件序号
        :return:
        """
        if isinstance(num, int):
            num = str(num)
        if os.path.exists(self._train_data_path) is False:
            os.mkdir(self._train_data_path)
        with open(self._train_data_path + 'train_data_' + num + '.pickle', "wb") as f:
            dump(zip(self._train_data, self._train_label), f)

    def _save_label_dict(self):
        if os.path.exists(self._dictionary_path) is False:
            os.mkdir(self._dictionary_path)
        with open(self._dictionary_path + 'label_dict.pickle', "wb") as f:
            dump(self._label_dict, f)

    def _save_num_dict(self):
        if os.path.exists(self._dictionary_path) is False:
            os.mkdir(self._dictionary_path)
        with open(self._dictionary_path + 'num_dict.pickle', "wb") as f:
            dump(self._num_dict, f)

    def _separation_word_and_label(self):
        self._character_tagging()
        # 建立两个词典
        self._label_dict = dict(zip(['S', 'B', 'M', 'E'], range(4)))
        self._save_label_dict()
        self._num_dict = {n: l for l, n in self._label_dict.items()}
        self._save_num_dict()
        with open(self._tag_corpus_path, encoding='utf-8') as f:
            lines = f.readlines()
            self._train_line = [[w[0] for w in line.split()] for line in lines]
            self._train_label_ = [[self._label_dict[w[2]] for w in line.split()] for line in lines]

    def _feat_context(self, sentence, word2idx, context=7):
        """

        :param sentence: 文本序列
        :param word2idx: 词典，通过词查找词序
        :param context: 取窗口长度为7
        :return:
        """
        predict_word_num = []
        # 找出每个词对应在词典当中得词序
        for w in sentence:  # 文本中的字如果在词典中则转为数字, 如果不在则设置为U对应得数字
            if w in word2idx:
                predict_word_num.append(word2idx[w])
            else:
                predict_word_num.append(word2idx[self.key_u])
        num = len(predict_word_num)  # 首尾padding
        pad = int((context - 1) * 0.5)
        for i in range(pad):
            # 头部添加个元素，尾部也添加个元素
            predict_word_num.insert(0, word2idx[self.key_p])
            predict_word_num.append(word2idx[self.key_p])
        x_data = []
        for i in range(num):
            x_data.append(predict_word_num[i:i + context])
        return x_data

    def _check_memory(self):
        """ 检查机器剩余内存

        :return:
        """
        if sys.platform == 'linux':
            with open('/proc/meminfo') as f:
                while True:
                    mem = f.readline()
                    if 'MemFree' in mem:
                        return int(self._pattern_num.findall(mem)[0])
        if sys.platform == 'win32':
            return psutil.virtual_memory().available / 1024 / 1024 / 1024

    def train_transform(self, word2idx, memory_size):
        """

        :param word2idx: 词典word2idx(通过词去查找词序)
        :param memory_size: 设置机器内存大小，当内存超过设定阈值时，将数据持久化到磁盘

        :return:
        """
        self._separation_word_and_label()
        num = 0
        for index, line in enumerate(self._train_line):
            self._train_data.extend(self._feat_context(line, word2idx))
            self._train_label.extend(self._train_label_[index])
            if self._check_memory() < memory_size:
                self.status = 1
                self._save_train_data(num)
                self._train_data.clear()
                self._train_label.clear()
            num += 1
        self._save_train_data('all')

    def predict_transform(self, input_txt, word2idx):
        """

        :param input_txt: 待分词文本
        :param word2idx: 词典word2idx(通过词去查找词序)
        :return:
        """
        return self._feat_context(input_txt, word2idx)

    def get_train_data(self):
        if self.status == 0:
            return self._train_data
        else:
            raise Exception("The train_data is so big, So this function doesn't to work!")

    def get_train_label(self):
        if self.status == 0:
            return self._train_label
        else:
            raise Exception("The train_label is so big, So this function doesn't to work!")
