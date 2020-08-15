#!/bin/bash
#description: RoBERT fine-tuning 使用mobile数据利用robert进行fine tuning

export BERT_BASE_DIR=./../albert_base
export DATA_DIR=./dat
export TRAINED_CLASSIFIER=./output
export MODEL_NAME=mobile_0_alberta_base

#win
python3 run_classifier_serving.py --task_name=setiment --do_train=true --do_eval=true --do_predict=False --data_dir=./dat --vocab_file=./../albert_base_zh/vocab.txt --bert_config_file=./../albert_base_zh/albert_config_base.json --init_checkpoint=./../albert_base_zh/bert_model.ckpt --max_seq_length=312 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=1.0 --output_dir=./output/mobile_0_alberta_base

#linux
python3 run_classifier_serving.py --task_name=setiment --do_train=true --do_eval=true --do_predict=False --data_dir=$DATA_DIR --vocab_file=$BERT_BASE_DIR/vocab_chinese.txt --bert_config_file=$BERT_BASE_DIR/albert_config.json --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=1.0 --output_dir=$TRAINED_CLASSIFIER/$MODEL_NAME