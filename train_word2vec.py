# -*- coding: utf-8 -*-
"""
该脚本用于将预料训练为词向量
"""
import os
import warnings
warnings.filterwarnings(
    action='ignore',
    category=UserWarning,
    module='gensim'
)
from w2v.word2vector import TrainWord2Vec

CORPUS_PATH = 'corpus/msr.txt'
if os.path.exists(CORPUS_PATH) is False:
    print("[WARN] The corpus file is not exist, please find it!")

TRAIN_WORD2VEC = TrainWord2Vec()


def train_w2v_run():
    TRAIN_WORD2VEC.load_file(CORPUS_PATH)
    TRAIN_WORD2VEC.train_word2vec(
        epochs=10,
        num_features=100,
        min_word_count=1,
        num_workers=4,
        context=5,
        sample=1e-3
    )
    TRAIN_WORD2VEC.transform_func()


if __name__ == '__main__':
    train_w2v_run()
    TRAIN_WORD2VEC.load_w2v()
    print(TRAIN_WORD2VEC.check_most_similar("上海"))
