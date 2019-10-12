#!/bin/bash
#chkconfig: 2345 80 90
#description: ����BERT����ģ�� 

echo 'start BERT classify server...'
rm -rf tmp*

export BERT_BASE_DIR=./chinese_roberta_zh_l12
export TRAINED_CLASSIFIER=./output
export EXP_NAME=mobile_0_roberta

bert-base-serving-start \
    -model_dir $TRAINED_CLASSIFIER/$EXP_NAME \
    -bert_model_dir $BERT_BASE_DIR \
    -model_pb_dir $TRAINED_CLASSIFIER/$EXP_NAME \
    -mode CLASS \
    -max_seq_len 128 \
    -port 5575 \
    -port_out 5576 \
    -device_map 1 
