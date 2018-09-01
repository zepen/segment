### 基于word2vec + LSTM的分词器

#### 项目环境
##### 开发环境: centos7, python3.5.0
##### 依赖python库: flask.1.0.2, keras.2.1.5, gensim.3.2.0, tensorflow.1.4.1, nltk.3.2.5, sklearn.0.19.1, pandas.0.21.0, h5py.2.8.0
##### (windos系统需要安装 psutil.5.4.0)

#### 脚本说明
##### 1.train_lstm.py 用于训练lstm模型
##### 2.train_word2vec.py 用于训练word2vec词向量
##### 3.serve_run.py 用于启动预测服务 

#### 执行顺序
###### step_1. 先执行train_word2vec.py 进行词向量训练
###### step_2. 再执行train_lstm.py 训练lstm模型
###### step_3. 最后启动 serve_run.py 服务
