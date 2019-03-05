from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # either 'voc' or 'inat' or 'inat-oneclass' or 'vott' or 'iwildcam'
    dataset = 'vott' 

    # Not defined her, passed in via train.sh
    inat_train_annotation = '' # .../iNat2017/train_2017_bboxes.json
    inat_val_annotation = '' # ../iNat2017/val_2017_bboxes.json
    inat_image_root = '' # ../iNat2017/images/
    voc_data_dir = '' # ../VOCdevkit/VOC2007/
    train_image_dir = '' # Only for VOTT format
    val_image_dir = '' # Only for VOTT format

    validate_only = False

    min_size = 600  # image resize
    max_size = 1000 # image resize
    num_workers = 4
    test_num_workers = 4

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.
    # Whether to weight background by 1 / number proposals in classification of RPN
    reduce_bg_weight = False

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 0.0003
    lr_schedule = [7,9]
    num_epochs = 13

    # visualization
    env = 'faster-rcnn'  # visdom env
    port = 8097
    plot_every = 40  # vis every N iter
    snapshot_every = 50000  # vis every N iter

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    batch_size = 1

    # not fully implemented yet
    use_cuda = True
    use_adam = False # Use Adam optimizer
    use_chainer = False # try match everything as chainer
    use_drop = False # use dropout in RoIHead
    # debug
    debug_file = '/tmp/debugf'

    test_num = 1000
    # model
    load_path = None

    caffe_pretrain = False # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = 'data/vgg16_caffe.pth'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
