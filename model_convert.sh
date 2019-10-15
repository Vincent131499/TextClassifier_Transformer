#!/bin/bash
#description: Convert model from ckpt format to pb format
#如果在模型文件中（$TRAINED_CLASSIFIER）存在label2id.pkl文件，此处可以不用指定num_labels参数

export BERT_BASE_DIR=./chinese_roberta_zh_l12
export TRAINED_CLASSIFIER=./output
export EXP_NAME=mobile_0_roberta_base_epoch1

python freeze_graph.py \
    -bert_model_dir $BERT_BASE_DIR \
    -model_dir $TRAINED_CLASSIFIER/$EXP_NAME \
    -max_seq_len 128