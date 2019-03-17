#!/bin/bash
for value in {1..20}
do
    checkp='./checkpoints/'$(ls checkpoints | tail -n 1)
    python train.py train \
--env='fasterrcnn-caffe' \
--plot-every=100 \
--caffe-pretrain  \
--train_annotation='/data/train_bboxes.json' \
--val_annotation='/data/val_bboxes.json' \
--image_root='/data/' \
--dataset='oneclass'  \
--load_path=$checkp
done
