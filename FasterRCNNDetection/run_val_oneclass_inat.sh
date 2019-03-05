#!/bin/bash
checkp='./checkpoints/'$(ls checkpoints | tail -n 1)
python train.py train \
--env='fasterrcnn-caffe' \
--plot-every=100 \
--caffe-pretrain  \
--inat_train_annotation='/datadrive/iNat2017/train_2017_bboxes.json' \
--inat_val_annotation='/datadrive/iNat2017/val_2017_bboxes.json' \
--inat_image_root='/datadrive/iNat2017/' \
--dataset='inat-oneclass' \
--load_path=$checkp \
--validate_only=True
