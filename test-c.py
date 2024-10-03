
import os, argparse, time, subprocess, io, shlex, pickle, pprint
import pandas as pd
import numpy as np
import tqdm
import fire
import torch


parser = argparse.ArgumentParser(description='Tiny ImageNet Testing - Corruptions')
## General parameters
parser.add_argument('--in_path', required=True,
                    help='path to Tiny ImageNet folder that contains test folders')
parser.add_argument('-o', '--output_path', default='model',
                    help='path for storing ')               
parser.add_argument('-restore_path', '--restore_path', default=None, type=str,
                    help='path of folder containing specific epoch file for restoring model training')
parser.add_argument('-model_path', '--model_path', default='model/Run1', type=str,
                    help='path of folder containing specific model Run for model testing')
parser.add_argument('--test_after_test', choices=[True, False], default=False, type=bool,
                    help='if true, process results after the test of all corruptions')
                    

## Parameters
parser.add_argument('--ngpus', default=1, type=int,
                    help='number of GPUs to use; 0 if you want to run on CPU')
parser.add_argument('-j', '--workers', default=8, type=int,  
                    help='number of data loading workers')
parser.add_argument('--batch_size', default=64, type=int,    
                    help='mini-batch size')

## Model parameters 
parser.add_argument('--torch_seed', default=0, type=int,
                    help='seed for weights initializations and torch RNG')  
parser.add_argument('--model_arch', choices=['voneresnet18', 'resnet18'], default='voneresnet18', 
                    help='model architecture to load')         
parser.add_argument('--normalization', choices=['vonenet', 'tinyimagenet'], default='vonenet',
                    help='image normalization to apply')
parser.add_argument('--visual_degrees', default=2, type=float,          
                    help='Field-of-View of the model in visual degrees')

## VOneBlock parameters
# Gabor filter bank
parser.add_argument('--stride', default=2, type=int,    
                    help='stride for the first convolution (Gabor Filter Bank)')
parser.add_argument('--ksize', default=31, type=int,    
                    help='kernel size for the first convolution (Gabor Filter Bank)')
parser.add_argument('--simple_channels', default=256, type=int,  
                    help='number of simple channels in V1 block')
parser.add_argument('--complex_channels', default=256, type=int, 
                    help='number of complex channels in V1 block')
parser.add_argument('--gabor_seed', default=0, type=int,
                    help='seed for gabor initialization')
parser.add_argument('--sf_corr', default=0.75, type=float,
                    help='')
parser.add_argument('--sf_max', default=11.5, type=float,          
                    help='')
parser.add_argument('--sf_min', default=0, type=float,
                    help='')
parser.add_argument('--rand_param', choices=[True, False], default=False, type=bool,
                    help='random gabor params')
parser.add_argument('--diff_n', choices=[True, False], default=False, type=bool,
                    help='different nx and ny')
parser.add_argument('--k_exc', default=25, type=float,
                    help='')

# Noise layer
parser.add_argument('--noise_mode', choices=['gaussian', 'neuronal', None],
                    default=None,
                    help='noise distribution')
parser.add_argument('--noise_scale', default=1, type=float,
                    help='noise scale factor')
parser.add_argument('--noise_level', default=1, type=float,
                    help='noise level')

FLAGS, FIRE_FLAGS = parser.parse_known_args()

def set_gpus(n=2):
    """
    Finds all GPUs on the system and restricts to n of them that have the most
    free memory.
    """
    if n > 0:
        gpus = subprocess.run(shlex.split(
            'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'), check=True,
            stdout=subprocess.PIPE).stdout
        gpus = pd.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
        gpus = gpus[gpus['memory.total [MiB]'] > 10000]  # only above 10 GB
        if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
            visible = [int(i)
                       for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
            gpus = gpus[gpus['index'].isin(visible)]
        gpus = gpus.sort_values(by='memory.free [MiB]', ascending=False)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # making sure GPUs are numbered the same way as in nvidia_smi
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            [str(i) for i in gpus['index'].iloc[:n]])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


if FLAGS.ngpus > 0:
    set_gpus(FLAGS.ngpus)


import torch
import torch.nn as nn
import torch.utils.model_zoo
import torchvision
from vonenet import get_model
import vonenet

torch.manual_seed(FLAGS.torch_seed)

torch.backends.cudnn.benchmark = True

if FLAGS.ngpus > 0:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = 'cpu'

categories = ["Noise", "Blur", "Weather", "Digital"]
corruptions = { "Gaussian" : "gaussian_noise",
               "Shot" : "shot_noise",
               "Impulse" : "impulse_noise",
               "DeFocus" : "defocus_blur",
               "Glass" : "glass_blur", 
               "Motion" : "motion_blur", 
               "Zoom" : "zoom_blur", 
               "Snow" : "snow", 
               "Frost" : "frost", 
               "Fog" : "fog", 
               "Brightness" : "brightness", 
               "Contrast" : "contrast", 
               "Elastic" : "elastic_transform", 
               "Pixelate" : "pixelate", 
               "JPEG" : "jpeg_compression"
                }

records = []

run = FLAGS.model_path[6:]
if os.path.isfile(os.path.join(FLAGS.output_path, 'results_corruptions_'+run+'.pkl')):
    results_old = pickle.load(open(os.path.join(FLAGS.output_path, 'results_corruptions_'+run+'.pkl'), 'rb'))
    for result in results_old:
        records.append(result)

if FLAGS.normalization == 'vonenet':
    print('VOneNet normalization')
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
elif FLAGS.normalization == 'tinyimagenet':
    print('Tiny Imagenet standard normalization')
    # From Freecodecamp-master utils.py
    norm_mean = [0.4914, 0.4822, 0.4465]
    norm_std = [0.2023, 0.1994, 0.2010]


def load_model():
    map_location = None if FLAGS.ngpus > 0 else 'cpu'
    print('Getting VOneNet')
    print(FLAGS.rand_param)
    model = get_model(map_location=map_location, model_arch=FLAGS.model_arch, pretrained=True, model_path=FLAGS.model_path,
                      visual_degrees=FLAGS.visual_degrees, stride=FLAGS.stride, ksize=FLAGS.ksize, diff_n=FLAGS.diff_n,
                      sf_corr=FLAGS.sf_corr, sf_max=FLAGS.sf_max, sf_min=FLAGS.sf_min, rand_param=FLAGS.rand_param,
                      gabor_seed=FLAGS.gabor_seed, simple_channels=FLAGS.simple_channels,
                      complex_channels=FLAGS.simple_channels, noise_mode=FLAGS.noise_mode,
                      noise_scale=FLAGS.noise_scale, noise_level=FLAGS.noise_level, k_exc=FLAGS.k_exc)

    if FLAGS.ngpus > 0 and torch.cuda.device_count() > 1:
        print('We have multiple GPUs detected')
        model = model.to(device)
    elif FLAGS.ngpus > 0 and torch.cuda.device_count() is 1:
        print('We run on GPU')
        model = model.to(device)
    else:
        print('No GPU detected!')
        model = model.module

    return model

def test():
    print("We perform test_after_test:", FLAGS.test_after_test)
    if not FLAGS.test_after_test:
        model = load_model()   

        for corr in corruptions:
            for i in range(5):
                in_path = os.path.join('../Tiny-ImageNet-C', corruptions[corr], str(i+1))
                print(in_path)
                tester = TinyImageNetTest(model, in_path)

                recent_time = time.time()
                results = {'meta': {'phase': 'test',
                                    'wall_time': recent_time,
                                    'category': '',
                                    'corr_type': corr,
                                    'corr_level': i+1 }
                        }

                if FLAGS.output_path is not None:
                    if not (os.path.isdir(FLAGS.output_path)):
                        os.mkdir(FLAGS.output_path)

                results[tester.name] = tester()

                if len(results) > 1:
                    pprint.pprint(results)
                    records.append(results)

        pickle.dump(records, open(os.path.join(FLAGS.output_path, 'results_corruptions_'+run+'.pkl'), 'wb'))

    else:   # test_after_test, postprocessing results
        results = pickle.load(open(os.path.join(FLAGS.output_path, 'results_corruptions_'+run+'.pkl'), 'rb'))
        
        accuracies = np.zeros(0,dtype=np.dtype([('corr_type', (np.str_, 20)), ('corr_level', np.int32), ('top1acc', np.float64)]))
        for result in results:
            meta = result['meta']
            res = result['test']
            corr_type = meta['corr_type']
            corr_lev = meta['corr_level']
            top1acc = res['top1']

            accuracies = np.append(accuracies, np.array([(corr_type, corr_lev, top1acc)], dtype=accuracies.dtype))

        average_acc = np.zeros(0, dtype=np.dtype([('corr_type', (np.str_, 20)), ('top1acc', np.float64)]))
        print(average_acc['top1acc']*100)



class TinyImageNetTest(object):

    def __init__(self, model, in_path):
        self.name = 'test'
        self.model = model
        self.data_loader = self.data(in_path)
        self.loss = nn.CrossEntropyLoss(size_average=False)
        self.loss = self.loss.to(device)

    def data(self, in_path):
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(in_path, ''),
            torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=norm_mean, std=norm_std),
            ]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  shuffle=False,
                                                  num_workers=FLAGS.workers,
                                                  pin_memory=True)

        return data_loader

    def __call__(self):
        self.model.eval()
        start = time.time()
        record = {'loss': 0, 'top1': 0, 'top5': 0}
        with torch.no_grad():
            for (inp, target) in tqdm.tqdm(self.data_loader, desc=self.name):
                target = target.to(device)
                output = self.model(inp)

                record['loss'] += self.loss(output, target).item()
                p1, p5 = accuracy(output, target, topk=(1, 5))
                record['top1'] += p1
                record['top5'] += p5

        for key in record:
            record[key] /= len(self.data_loader.dataset.samples)
        record['dur'] = (time.time() - start) / len(self.data_loader)

        return record


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res


if __name__ == '__main__':
    fire.Fire(command=FIRE_FLAGS)
