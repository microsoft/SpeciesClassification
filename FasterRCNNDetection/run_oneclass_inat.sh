#!/bin/bash
for value in {1..20}
do
    checkp='./checkpoints/'$(ls checkpoints | tail -n 1)
    python train.py train \
--env='fasterrcnn-caffe' \
--plot-every=100 \
--caffe-pretrain  \
--inat_train_annotation='/data/data/iNat2017/train_2017_bboxes.json' \
--inat_val_annotation='/data/data/iNat2017/val_2017_bboxes.json' \
--inat_image_root='/data/data/iNat2017/' \
--dataset='inat-oneclass'  \
--load_path=$checkp
done
