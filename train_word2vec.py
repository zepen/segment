# -*- coding: utf-8 -*-
"""
该脚本用于将预料训练为词向量
"""
from w2v.word2vector import TrainWord2Vec


def train_w2v_run():
    train_word2vec = TrainWord2Vec()
    train_word2vec.load_file('corpus/msr.txt')
    train_word2vec.train_word2vec(
        epochs=20,
        num_features=100,
        min_word_count=1,
        num_workers=4,
        context=4,
        sample=1e-53
    )
    train_word2vec.transform_func()


if __name__ == '__main__':
    train_w2v_run()
