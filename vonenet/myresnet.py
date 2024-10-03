
from collections import OrderedDict
from torch import nn
from .back_ends import ResNet, BasicBlock


def myResNet(k_exc=25, model_arch='resnet18', image_size=64, visual_degrees=2, ksize=31, stride=2):

    arch_params = {'k_exc': k_exc, 'arch': model_arch, 'ksize': ksize, 'stride': stride}


    if model_arch:
        if model_arch.lower() == 'resnet18':
            print('Model: ', 'Resnet18')
            model_back_end = ResNet(block=BasicBlock, num_classes=200, layers=[2, 2, 2, 2])

        model = nn.Sequential(OrderedDict([
            ('model', model_back_end),
        ]))

    model.image_size = image_size
    model.visual_degrees = visual_degrees
    model.arch_params = arch_params

    return model
