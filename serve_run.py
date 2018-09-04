import json
import os
import time
from flask import Flask
from flask import request, Response, render_template
from lstm.lstm_net import LongShortTMNet
from lstm.processing import DataProcessing
from pickle import load

app = Flask(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

INIT_WEIGHT_WV_PATH = 'model/init_weight_wv.pickle'

VOCAB_DICT_PATH = 'dictionary/word2idx.pickle'
LABEL_DICT_PATH = 'dictionary/label_dict.pickle'
NUM_DICT_PATH = 'dictionary/num_dict.pickle'

# 全局加载词典
if os.path.exists(VOCAB_DICT_PATH):
    with open(VOCAB_DICT_PATH, 'rb') as f:
        vocab_dict = load(f)
    print("[INFO] The vocab_dict is load!")
else:
    raise Exception("[ERROR] The word2idx.pickle is not exist!")

if os.path.exists(LABEL_DICT_PATH):
    with open(LABEL_DICT_PATH, 'rb') as f:
        label_dict = load(f)
    print("[INFO] The label_dict is load!")
else:
    raise Exception("[ERROR] The label_dict.pickle is not exist!")

if os.path.exists(NUM_DICT_PATH):
    with open(NUM_DICT_PATH, 'rb') as f:
        num_dict = load(f)
    print("[INFO] The num_dict is load!")
else:
    raise Exception("[ERROR] The num_dict.pickle is not exist!")

# 全局加载模型
data_processing = DataProcessing()
lstm_net = LongShortTMNet()
lstm_net.model_load('lstm_model')


@app.route('/')
def main():
    return 'welcome to use!'


@app.route('/cut_word', methods=['POST', 'GET'])
def cut_word():
    try:
        if request.method == 'POST':
            if len(request.form) != 0:
                context = request.form['sequences']
                start_time = time.time()
                x_data = data_processing.predict_transform(context, vocab_dict)
                result = lstm_net.cut_word(x_data, context, label_dict, num_dict)
                end_time = time.time()
                print("Cost time is: ", end_time - start_time)
                return render_template('predict_message_test.html', result=result)
            else:
                data_warn = {"warning": "No words to cut!"}
                return Response(json.dumps(data_warn))
        else:
            return render_template('predict_message_test.html')
    except Exception as e:
        print(e)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5555)
