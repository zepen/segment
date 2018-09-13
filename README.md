### 基于 word2vec + LSTM 的分词器

#### 一、项目环境
##### 开发环境: centos7, python3.5.0
##### 依赖python库: flask.1.0.2, keras.2.1.5, gensim.3.2.0, tensorflow.1.4.1, nltk.3.2.5, sklearn.0.19.1, pandas.0.21.0, h5py.2.8.0 gevent.1.3.6
##### (windos系统需要安装 psutil.5.4.0;注意keras和tensorflow版本对应,版本不对可能会无法正常运行)

#### 二、脚本说明
##### 1.train_lstm.py 用于训练lstm模型
##### 2.train_word2vec.py 用于训练word2vec词向量
##### 3.serve_run.py 用于启动预测服务 

#### 三、执行顺序
###### step_1. 先执行train_word2vec.py 进行词向量训练
###### step_2. 再执行train_lstm.py 训练lstm模型
###### step_3. 最后启动 serve_run.py 服务

#### 四、模型结构
###### 通过调用get_model_structureh函数查看

#### 五、超参说明
| 超参数 | 参数说明 |
| ---- | ---- |
| word_dim | 词向量长度 |
| max_len | 最大窗口 |
| hidden_units | 隐藏节点数 |
| batch_size | 批量训练数据大小 |
| epochs | 迭代次数 |
| val_loss | 验证集上损失函数值 |
| val_acc | 验证集上模型准确率 |

#### 六、模型评估
| word_dim| max_len | hidden_units | batch_size | epochs | val_loss| val_acc |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 100 | 7 | 100 | 128 | 20 | 0.2038 | 92.7% |
| 100 | 7 | 512 | 128 | 30 | 0.1003 | 96.5% |
#### 七、References
##### [1]郑捷，NLP汉语自然语言处理原理与实践[M].北京：电子工业出版社，2017年1月.