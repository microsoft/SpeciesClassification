python train.py --data_root /data/animals2/iNat2017_extended/ \
--model_type resnext101 \
--image_size 560 \
--resume models/resnext-560-80.1/model_best.pth.tar \
--evaluate \
--train_file trainval2017.json \
--val_file minival2017.json
