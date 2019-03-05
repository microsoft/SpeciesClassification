# By using the software, you accept these terms. If you do not accept them, do not use the software. 
# The software is provided “as-is”.  You bear the risk of using it.  Microsoft gives no express
# warranties, guarantees or conditions.

python train.py train \
--env='fasterrcnn-caffe' \
--plot-every=100 \
--caffe-pretrain  \
--dataset='vott' \
--train_image_dir='/data/plastic/train-filtered' \
--val_image_dir='/data/plastic/test-filtered/' \
--num_epochs=2000 \
--lr_schedule=[1500] \
--plot_every=20 \
--max_size=1920 \
--min_size=1080 \
--num_workers=8 \
--reduce_bg_weight=False \
--lr=0.001 \
--load_path=checkpoints/fasterrcnn_10041721_0.19082711229683977

# Make sure you set in train.py: 
#       faster_rcnn = FasterRCNNVGG16(n_fg_class=dataset.get_class_count(), anchor_scales=[1], ratios=[1])
