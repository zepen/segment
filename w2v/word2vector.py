"""
定义训练词向量类
"""
import codecs
import pandas as pd
import numpy as np
from nltk.probability import FreqDist
from nltk.text import Text
from gensim.models import word2vec
from pickle import dump


class TrainWord2Vec(object):

    def __init__(self):
        self.corpus_path="corpus/msr.txt"
        self.input_text = None
        self.freq_df = None
        self._w2v = None
        self._w2v_path = "model/word_vector.bin"
        self._word2idx = {}
        self._word2idx_path = 'dictionary/word2idx.pickle'
        self._idx2word = {}
        self._idx2word_path = 'dictionary/idx2word.pickle'
        self._init_weight_wv = []
        self._init_weight_wv_path = 'model/init_weight_wv.pickle'

    def __str__(self):
        return "This is train word2vector!"

    def load_file(self, input_file):
        """

        :param input_file: 输入预料文件
        :return:
        """
        input_data = codecs.open(input_file, 'r', 'utf-8')
        try:
            self.input_text = input_data.read()
            try:
                input_data.close()
            except Exception as e:
                print('[word2vector_close_input_data]' + str(e))
        except Exception as e:
            print("[load_file]" + str(e))

    def _freq_func(self):
        if self.input_text is not None:
            input_txt = [w for w in self.input_text.split()]
        else:
            raise Exception('The attr input_text object is None!')
        corpus = Text(input_txt)
        f_dist = FreqDist(corpus)
        w, v = zip(*f_dist.items())
        self.freq_df = pd.DataFrame({'word': w, 'freq': v})
        self.freq_df.sort_values('freq', ascending=False, inplace=True)
        self.freq_df['idx'] = np.arange(len(v))

    def _create_freq_dict(self):
        self._freq_func()
        zip_ = zip(self.freq_df.word, self.freq_df.idx)
        self._word2idx = dict((c, i) for c, i in zip_)
        self._idx2word = dict((i, c) for c, i in zip_)

    def train_word2vec(self, **kwargs):
        corpus = [line.split() for line in self.input_text.split('\n') if line != '']
        self._w2v = word2vec.Word2Vec(
            workers=kwargs['num_workers'],
            sample=kwargs['sample'],
            size=kwargs['num_features'],
            min_count=kwargs['min_word_count'],
            window=kwargs['context']
        )
        np.random.shuffle(corpus)
        self._w2v.build_vocab(corpus)
        print("Train is begin...")
        for epoch in range(kwargs['epochs']):
            print('epoch' + str(epoch))
            np.random.shuffle(corpus)
            self._w2v.train(corpus, total_examples=self._w2v.corpus_count, epochs=self._w2v.iter)
            self._w2v.alpha *= 0.9
            self._w2v.min_alpha = self._w2v.alpha
        print("w2v train is done...")
        self._save_w2v()

    def transform_func(self):
        """ 定义 'U'为未登陆新字, 'P'为两头padding用途, 并增加两个相应的向量表示
        :return:
        """
        self._create_freq_dict()
        if self._w2v is None:
            raise Exception('The w2v object is None!')
        for i in range(len(self._idx2word)):
            self._init_weight_wv.append(self._w2v[self._idx2word[i]])
        char_num = len(self._init_weight_wv)
        # idx2word
        self._idx2word[char_num] = u'U'
        self._idx2word[char_num + 1] = u'P'
        self._save_idx2word()
        # word2idx
        self._word2idx[u'U']=char_num
        self._word2idx[u'P'] = char_num + 1
        self._save_word2idx()
        # init_weight_wv
        self._init_weight_wv.append(np.random.randn(100, ))
        self._init_weight_wv.append(np.zeros(100, ))
        self._save_init_weight()

    def _save_w2v(self):
        if self._w2v is not None:
            self._w2v.save(self._w2v_path)
        else:
            raise Exception("The w2v object is None!")

    def _save_init_weight(self):
        if self._init_weight_wv is not None:
            with open(self._init_weight_wv_path, "wb") as f:
                dump(self._init_weight_wv, f)
        else:
            raise Exception("The init_weight_wv object is None!")

    def _save_idx2word(self):
        if self._idx2word is not None:
            with open(self._idx2word_path, "wb") as f:
                dump(self._idx2word, f)
        else:
            raise Exception("The idx2word object is None!")

    def _save_word2idx(self):
        if self._word2idx is not None:
            with open(self._word2idx_path, "wb") as f:
                dump(self._word2idx, f)
        else:
            raise Exception("The word2idx object is None!")

    def get_word2vec(self):
        return self._w2v

    def get_init_weight(self):
        return self._init_weight_wv

    def get_idx2word(self):
        return self._idx2word

    def get_word2idx(self):
        return self._word2idx
