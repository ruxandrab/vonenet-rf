import torch
import torch.nn as nn
import os

from torch.nn import Module

FILE_WEIGHTS = {'voneresnet18': 'epoch_60.pth.tar',
                'resnet18': 'epoch_60.pth.tar'}


class Wrapper(Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.module = model


def get_model(model_arch='resnet18', pretrained=True, map_location='cpu', model_path='model/Run1', **kwargs):
    """
    Returns a VOneNet model.
    Select pretrained=True for returning one of the 4 pretrained models.
    model_arch: string with identifier to choose the architecture of the back-end (resnet18)
    """
    if pretrained and model_arch:
        if model_arch=='voneresnet18' or model_arch=='resnet18':
            home_dir = '.'
            vonenet_dir = os.path.join(home_dir, model_path)
            weightsdir_path = os.path.join(vonenet_dir, FILE_WEIGHTS[model_arch.lower()])

        ckpt_data = torch.load(weightsdir_path, map_location=map_location)

        stride = ckpt_data['flags']['stride']
        simple_channels = ckpt_data['flags']['simple_channels']
        complex_channels = ckpt_data['flags']['complex_channels']
        k_exc = ckpt_data['flags']['k_exc']
        rand_param = ckpt_data['flags']['rand_param']
        diff_n = ckpt_data['flags']['diff_n']
        gabor_seed = ckpt_data['flags']['gabor_seed']

        sf_corr=ckpt_data['flags']['sf_corr']
        sf_max=ckpt_data['flags']['sf_max']
        sf_min=ckpt_data['flags']['sf_min']
        visual_degrees=ckpt_data['flags']['visual_degrees']
        ksize=ckpt_data['flags']['ksize']

        noise_mode = ckpt_data['flags']['noise_mode']
        noise_scale = ckpt_data['flags']['noise_scale']
        noise_level = ckpt_data['flags']['noise_level']

        if model_arch.lower() == 'voneresnet18' or model_arch.lower() == 'resnet18':
            model_id = model_arch
        
        if model_arch=='resnet18':
            model = globals()[f'myResNet'](model_arch=model_id, stride=stride, k_exc=k_exc,
                                        simple_channels=simple_channels, complex_channels=complex_channels,
                                        noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level)
        else:
            model = globals()[f'VOneNet'](model_arch=model_id, stride=stride, k_exc=k_exc, 
                                        sf_corr=sf_corr, sf_max=sf_max, sf_min=sf_min,
                                        visual_degrees=visual_degrees, ksize=ksize,
                                        rand_param=rand_param, diff_n=diff_n, gabor_seed=gabor_seed,
                                        simple_channels=simple_channels, complex_channels=complex_channels,
                                        noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level)

        if model_arch.lower() == 'voneresnet18' or model_arch.lower() == 'resnet18':
            model = Wrapper(model)
            model.load_state_dict(ckpt_data['state_dict'], strict=False)  
            model = model.module

        model = nn.DataParallel(model)
    else:
        if model_arch=='resnet18':
            model = globals()[f'myResNet'](model_arch=model_arch, **kwargs)
        else:
            model = globals()[f'VOneNet'](model_arch=model_arch, **kwargs)
        model = nn.DataParallel(model)

    model.to(map_location)
    return model

