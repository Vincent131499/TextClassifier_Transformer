# -*- coding: utf-8 -*-
"""
   File Name：     api_service_flask
   Description :  api请求服务端，并基于flask提供接口待查询
   Author :       逸轩
   date：          2019/10/12

"""

import json
import re
import time
from bert_base.client import BertClient
from flask import Flask, request
from flask_cors import CORS

flaskAPP = Flask(import_name=__name__)
CORS(flaskAPP, supports_credentials=True)

bc = BertClient(ip='192.168.9.23', port=5575, port_out=5576, show_server_config=False, check_version=False, check_length=False, mode='CLASS')
print('BertClient连接成功')

# 切分句子
def cut_sent(txt):
    # 先预处理去空格等
    txt = re.sub('([　 \t]+)', r" ", txt)  # blank word
    txt = txt.rstrip()  # 段尾如果有多余的\n就去掉它
    nlist = txt.split("\n")
    nnlist = [x for x in nlist if x.strip() != '']  # 过滤掉空行
    return nnlist


# 对句子列表进行预测识别
def class_pred(list_text):
    # 文本拆分成句子
    # list_text = cut_sent(text)
    print("total setance: %d" % (len(list_text)))
    # with BertClient(ip='192.168.9.23', port=5575, port_out=5576, show_server_config=False, check_version=False,
    #                 check_length=False, mode='CLASS') as bc:
    start_t = time.perf_counter()
    rst = bc.encode(list_text)
    print('result:', rst)
    print('time used:{}'.format(time.perf_counter() - start_t))
    # 返回结构为：
    # rst: [{'pred_label': ['0', '1', '0'], 'score': [0.9983683228492737, 0.9988993406295776, 0.9997349381446838]}]
    # 抽取出标注结果
    pred_label = rst[0]["pred_label"]
    result_txt = [[pred_label[i], list_text[i]] for i in range(len(pred_label))]
    return result_txt

@flaskAPP.route('/predict_online', methods=['GET', 'POST'])
def predict_online():
    text = request.args.get('text')
    print('服务器接收到字段：')
    print('text：', text)
    print('==============================')
    lstseg = cut_sent(text)
    print('-' * 30)
    print('结果,共%d个句子:' % (len(lstseg)))
    # for x in lstseg:
    #     print("第%d句：【 %s】" % (lstseg.index(x), x))
    print('-' * 30)
    # if request.method == 'POST' or 1:
    #     res['result'] = class_pred(lstseg)
    result = class_pred(lstseg)
    new_res_list = []
    for term in result:
        if term[0] == '1':
            label = '好评'
        if term[0] == '0':
            label = '中评'
        if term[0] == '-1':
            label = '差评'
        new_res_list.append([label, term[1]])
    new_res = {'result': new_res_list}
    # print('result:%s' % str(res))
    print('result:%s' % str(new_res))
    # return jsonify(res)
    return json.dumps(new_res, ensure_ascii=False)

flaskAPP.run(host='192.168.9.23', port=8910, debug=True)

