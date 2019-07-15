import argparse
from models import *
import torch.nn as nn
import torch
import data_loader

class ModelWrapper(nn.Module):
    def __init__(self, model, mean = torch.cuda.FloatTensor([0.5]), std = torch.cuda.FloatTensor([0.5])):
        super(ModelWrapper, self).__init__()
        self.max_val = torch.cuda.FloatTensor([255])
        self.model = model
        self.mean = mean
        self.std = std
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Transform uint8 values to [-1, 1]
        x = (x / self.max_val - self.mean) / self.std
        x = self.model(x)
        x = self.softmax(x)
        return x

def main():
    parser = argparse.ArgumentParser(description='PyTorch model ONNX export.')

    parser.add_argument('--model_type', default=ModelType.resnext101, 
                        metavar='ARCH', type=ModelType.from_string, choices=list(ModelType),
                        help='model architecture: ' + ' | '.join([m.name for m in ModelType]) +
                        ' (default: resnext101)')
    parser.add_argument('--image_size', default=224, nargs='+',
                        type=int, metavar='RESOLUTION', help='The side length of the CNN input image ' + \
                        '(default: 448). For ensembles, provide one resolution for each network.')
    parser.add_argument('--weights_path', default=None, nargs='+',
                        type=str, metavar='PATH', help='Path to a checkpoint to load from. Can ' + \
                        'be multiple checkpoints when combining multiple models to an ensemble from two different checkpoints.')
    parser.add_argument('--num_classes', default=5089,
                        type=int, metavar='NUM_CLASSES', help='Number of output elements of the network.')
    parser.add_argument('--output_prefix', default='onnx_export',
                        type=str, metavar='OUTPUT', help='Prefix for all output files. It is possible ' + \
                        'to provide a full path prior to the prefix, e.g. /path/to/output_prefix')
    parser.add_argument('--class_mask', default=None, metavar='CLASS_MASK.txt',
                        type=str, help='You can pass the path path to a text file containing a list of 0 and 1 here. The list ' + \
                        'should have the same number of outputs as the number of predicted classes. If the i-th entry in the list is 0, ' + \
                        'then we will remove this class from the exported onnx model.')
    args = parser.parse_args()


    model = ClassificationModel(args.weights_path, args.image_size, True, args.model_type)
    if args.class_mask is not None:
        selected_classes = np.loadtxt(args.class_mask).astype(np.bool)
        old_fc = model.state_dict()['model.last_linear.weight'].detach().cpu().numpy()[selected_classes]
        old_bias = model.state_dict()['model.last_linear.bias'].detach().cpu().numpy()[selected_classes]
        model.model.last_linear = nn.Linear(model.model.last_linear.in_features, np.sum(selected_classes)).cuda()
        model.state_dict()['model.last_linear.weight'].data.copy_(torch.from_numpy(old_fc))
        model.state_dict()['model.last_linear.bias'].data.copy_(torch.from_numpy(old_bias))
    model.eval()

    def disable_ceil_mode(m):
        for mc in list(m.children()):
            if type(mc) in [torch.nn.MaxPool2d, torch.nn.AvgPool2d]:
                mc.ceil_mode = False
            disable_ceil_mode(mc)
    disable_ceil_mode(model)

    model_wrapper = ModelWrapper(model)

    dummy_input = torch.autograd.Variable(torch.randn(1, 3, max(args.image_size), max(args.image_size))).cuda()
    print(model_wrapper(dummy_input))

    torch.onnx.export(model_wrapper, dummy_input, args.output_prefix + '_model.onnx', verbose=False)
    data_loader.save_model({
        'epoch': -1,
        'args': args,
        'state_dict': model.state_dict(),
        'best_prec1': -1,
        'best_prec3': -1,
        'best_prec5': -1,
        'classnames' : model.get_classnames(),
        'num_classes' : args.num_classes,
        'model_type' : args.model_type,
    }, False, filename = args.output_prefix + '_model.pytorch')

    # Export class names
    classname_list = [model.get_classnames()[cid] for cid in range(args.num_classes)]
    with open(args.output_prefix + '_classes.txt', 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(classname_list))


if __name__ == '__main__':
    main()
