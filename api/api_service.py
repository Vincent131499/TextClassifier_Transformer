# -*- coding: utf-8 -*-
"""
   File Name：     api_service_new
   Description :  api客户端请求服务器，返回标签
   Author :       逸轩
   date：          2019/10/12

"""

import json
import re
import time
from bert_base.client import BertClient

bc = BertClient(ip='192.168.9.23', port=5575, port_out=5576, show_server_config=False, check_version=False, check_length=False, mode='CLASS')
print('BertClient连接成功')

# 切分句子
def cut_sent(txt):
    # 先预处理去空格等
    txt = re.sub('([　 \t]+)', r" ", txt)  # blank word
    txt = txt.rstrip()  # 段尾如果有多余的\n就去掉它
    nlist = txt.split("；")
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
    # print('result:', rst)
    print('time used:{}s'.format(time.perf_counter() - start_t))
    # 返回结构为：
    # rst: [{'pred_label': ['0', '1', '0'], 'score': [0.9983683228492737, 0.9988993406295776, 0.9997349381446838]}]
    # 抽取出标注结果
    pred_label = rst[0]["pred_label"]
    result_txt = [[pred_label[i], list_text[i]] for i in range(len(pred_label))]
    return result_txt

if __name__ == '__main__':
    while True:
        text = input(r'请输入句子（多个句子请用；分隔）：')
        list_text = cut_sent(text)
        # print(list_text)
        result = class_pred(list_text)
        print(result)
