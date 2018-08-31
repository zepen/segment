import json
import time
from flask import Flask
from flask import request, Response, render_template
from lstm.lstm_net import LongShortTMNet
from lstm.processing import DataProcessing
from pickle import load

app = Flask(__name__)
# 全局加载词典
with open('dictionary/word2idx.pickle', 'rb') as f:
    word2idx = load(f)

with open('dictionary/label_dict.pickle', 'rb') as f:
    label_dict = load(f)

with open('dictionary/num_dict.pickle', 'rb') as f:
    num_dict = load(f)

# 全局加载模型
data_processing = DataProcessing()
lstm_net = LongShortTMNet()
lstm_net.model_load('lstm_model.h5')


@app.route('/')
def main():
    return 'welcome to use!'


@app.route('/cut_word', methods=['POST', 'GET'])
def cut_word():
    try:
        if request.method == 'POST':
            if len(request.form) != 0:
                context = request.form['sequences']
                # context = json_to_dict.get("context").encode("utf-8")
                start_time = time.time()
                x_data = data_processing.predict_transform(context, word2idx)
                result = lstm_net.cut_word(x_data, context, label_dict, num_dict)
                end_time = time.time()
                print("Cost time is: ", end_time - start_time)
                return render_template('predict_message_test.html', result=result)
            else:
                data_warn = {"warning": "No words to cut!"}
                return Response(json.dumps(data_warn))
        else:
            # method_warn = {"warning": "request method is wrong!"}
            return render_template('predict_message_test.html')
    except Exception as e:
        print(e)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555)
