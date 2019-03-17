#!/bin/bash
checkp='./checkpoints/'$(ls checkpoints | tail -n 1)
python train.py train \
--env='fasterrcnn-caffe' \
--plot-every=100 \
--caffe-pretrain  \
--train_annotation='/datadrive/train_bboxes.json' \
--val_annotation='/datadrive/val_bboxes.json' \
--image_root='/datadrive/' \
--dataset='oneclass' \
--load_path=$checkp \
--validate_only=True
