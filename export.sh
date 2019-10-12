#!/bin/bash
#description: BERT fine-tuning

export BERT_BASE_DIR=./chinese_roberta_zh_l12
export DATA_DIR=./dat
export TRAINED_CLASSIFIER=./output
export MODEL_NAME=mobile_0_roberta

python run_classifier_serving.py \
  --task_name=setiment \
  --do_train=False \
  --do_eval=False \
  --do_predict=True \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=2.0 \
  --output_dir=$TRAINED_CLASSIFIER/$MODEL_NAME \
  --do_export=True \
  --export_dir=exported