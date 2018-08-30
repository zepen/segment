### 基于word2ve + LSTM的分词器
##### 1.train_lstm.py 用于训练lstm模型
##### 2.train_word2vec.py 用于训练word2vec词向量
##### 3.serve_run.py 用于启动预测服务 

### 执行顺序
###### step_1. 先执行train_word2vec.py 进行词向量训练
###### step_2. 再执行train_lstm.py 训练lstm模型
###### step_3. 最后启动 serve_run.py 服务
