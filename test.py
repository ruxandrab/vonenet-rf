
import os, argparse, time, subprocess, io, shlex, pickle, pprint
import pandas as pd
import numpy as np
import tqdm
import fire
from matplotlib.pylab import plt
import torch


parser = argparse.ArgumentParser(description='Tiny ImageNet Testing')
## General parameters
parser.add_argument('--in_path', required=True,
                    help='path to Tiny ImageNet folder that contains test folders')
parser.add_argument('-o', '--output_path', default='model',
                    help='path for storing ')               
parser.add_argument('-restore_path', '--restore_path', default=None, type=str,
                    help='path of folder containing specific epoch file for restoring model training')
parser.add_argument('-model_path', '--model_path', default='model/Run1', type=str,
                    help='path of folder containing specific model Run for model testing')
parser.add_argument('-results_postproc_filename', '--results_postproc_filename', default='results_postproc.pkl', type=str,
                    help='name of file to store the results of model testing')

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
                    help='back-end model architecture to load')         
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

# Plot params
parser.add_argument('--plot_activ_VOne', choices=[True, False], default=True, type=bool,
                    help='Plot activations after the VOneBlock')
parser.add_argument('--plot_weights', choices=[True, False], default=True, type=bool,
                    help='Plot weights')
parser.add_argument('--show_or_save_figs', choices=['show', 'save', None], default=None,
                    help='Show or save figures')

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

records = []

if os.path.isfile(os.path.join(FLAGS.output_path, FLAGS.results_postproc_filename)):
    results_old = pickle.load(open(os.path.join(FLAGS.output_path, FLAGS.results_postproc_filename), 'rb'))
    for result in results_old:
        records.append(result)

sch = FLAGS.simple_channels
cch = FLAGS.complex_channels
allch = sch + cch


if FLAGS.normalization == 'vonenet':
    print('VOneNet normalization')
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
elif FLAGS.normalization == 'tinyimagenet':
    print('Tiny Imagenet standard normalization')
    norm_mean = [0.4914, 0.4822, 0.4465]
    norm_std = [0.2023, 0.1994, 0.2010]


def load_model():
    map_location = None if FLAGS.ngpus > 0 else 'cpu'
    print('Getting VOneNet')
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


def make_colormap_no_and_mean(qois,qoic, oxs,oys, Xs,Ys, oxc,oyc, Xc,Yc, thresh, xlabel, ylabel, title, figname):
    ## Simple cells
    #
    # Bin quantity of interest according to their 0x
    binned_oxs = [qois[np.where((oxs > xlow) & (oxs <= xhigh))] 
                    for xlow, xhigh in zip(Xs[:-1], Xs[1:])]
    # Bin quantity of interest  according to their 0y
    binned_oys = [qois[np.where((oys > ylow) & (oys <= yhigh))] 
                    for ylow, yhigh in zip(Ys[:-1], Ys[1:])]
    
    binned_combs = []
    binned_comb_nos = np.zeros((len(binned_oxs),len(binned_oys)))
    binned_comb_means = np.zeros((len(binned_oxs),len(binned_oys)))
    # Get the bins intersections, the number of items and the means inside each bin
    for i in range(len(binned_oxs)):
        for j in range(len(binned_oys)):
            inters = np.intersect1d(binned_oxs[i], binned_oys[j])
            if len(inters)>=thresh:
                binned_combs.append(inters)
                binned_comb_nos[i][j] = len(inters)
                binned_comb_means[i][j] = np.mean(inters)

    ## Complex cells
    #
    binned_oxc = [qoic[np.where((oxc > xlow) & (oxc <= xhigh))] 
                         for xlow, xhigh in zip(Xc[:-1], Xc[1:])]
    binned_oyc = [qoic[np.where((oyc > ylow) & (oyc <= yhigh))] 
                    for ylow, yhigh in zip(Yc[:-1], Yc[1:])]

    binned_combc = []
    binned_comb_noc = np.zeros((len(binned_oxc),len(binned_oyc)))
    binned_comb_meanc = np.zeros((len(binned_oxc),len(binned_oyc)))
    for i in range(len(binned_oxc)):
        for j in range(len(binned_oyc)):
            interc = np.intersect1d(binned_oxc[i], binned_oyc[j])
            if len(interc)>=thresh:
                binned_combc.append(interc)
                binned_comb_noc[i][j] = len(interc)
                binned_comb_meanc[i][j] = np.mean(interc)

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))

    # Simple cells, no of neurons
    im0 = ax0.imshow(binned_comb_nos.T, origin='lower')
    fig.colorbar(im0, ax=ax0)
    ax0.set(xlabel=xlabel, ylabel=ylabel)  
    ax0.set_xticks(np.arange(-.5, len(Xs)-1, 1))
    ax0.set_yticks(np.arange(-.5, len(Ys)-1, 1))
    ax0.set_xticklabels(np.round(np.log2(Xs),2))
    ax0.set_yticklabels(np.round(np.log10(Ys),2))
    ax0.set_aspect('equal', adjustable='box')

    # Complex cells, no of neurons
    im1 = ax1.imshow(binned_comb_noc.T, origin='lower')
    ax1.set(xlabel=xlabel, ylabel=ylabel)
    ax1.set_xticks(np.arange(-.5, len(Xc)-1, 1))
    ax1.set_yticks(np.arange(-.5, len(Yc)-1, 1))
    ax1.set_xticklabels(np.round(np.log2(Xc),2))
    ax1.set_yticklabels(np.round(np.log10(Yc),2)) 
    fig.colorbar(im1, ax=ax1)
    ax1.set_aspect('equal', adjustable='box')
    
    # Simple cells, mean
    im2 = ax2.imshow(binned_comb_means.T, origin='lower')
    fig.colorbar(im2, ax=ax2)
    ax2.set(xlabel=xlabel, ylabel=ylabel)  
    ax2.set_xticks(np.arange(-.5, len(Xs)-1, 1))
    ax2.set_yticks(np.arange(-.5, len(Ys)-1, 1))
    ax2.set_xticklabels(np.round(np.log2(Xs),2))
    ax2.set_yticklabels(np.round(np.log10(Ys),2))

    # Complex cells, mean
    im3 = ax3.imshow(binned_comb_meanc.T, origin='lower')
    ax3.set(xlabel=xlabel, ylabel=ylabel)
    ax3.set_xticks(np.arange(-.5, len(Xc)-1, 1))
    ax3.set_yticks(np.arange(-.5, len(Yc)-1, 1))
    ax3.set_xticklabels(np.round(np.log2(Xc),2))
    ax3.set_yticklabels(np.round(np.log10(Yc),2)) 
    fig.colorbar(im3, ax=ax3)

    fig.suptitle(title)
    if FLAGS.show_or_save_figs == 'show':
        plt.show()
    elif FLAGS.show_or_save_figs == 'save':
        plt.savefig(os.path.join(FLAGS.model_path, FLAGS.model_arch+figname))
    plt.clf()

    results = {'gabor_seed': str(FLAGS.gabor_seed), 
                'cell_type': 'simple and complex', 
                'type': figname,
                'qois': qois,
                'qoic': qoic,
                'oxs': oxs,
                'oys': oys,
                'oxc': oxc,
                'oyc': oyc,
                'Xs': Xs,
                'Ys': Ys,
                'Xc': Xc,
                'Yc': Yc,
                'xlabel': xlabel, 
                'ylabel': ylabel, 
                'title': title
               }
    return results



def test():

    model = load_model()

    if FLAGS.model_arch!='resnet18':
        if FLAGS.plot_weights == True:
            print('Get weights of the bottleneck layer')
            wb = model.module.bottleneck.weight.transpose(0,1)
            wb = wb.reshape((allch,64))

            wbs = abs(wb[0:sch,:])
            wbc = abs(wb[sch:,:])
            wbs_mean = wbs.mean(dim=1)   
            wbc_mean = wbc.mean(dim=1)   
            wbs_means = wbs_mean.cpu().data.numpy()
            wbc_means = wbc_mean.cpu().data.numpy()
            wb_means = abs(wb).mean(dim=1)
            wb_means = wb_means.cpu().data.numpy()
            
            ## Process sf
            sfs = model.module.vone_block.sf[0:sch]
            sfc = model.module.vone_block.sf[sch:]

            ## Process sigxy
            sigxs = model.module.vone_block.sigx[0:sch]
            sigxc = model.module.vone_block.sigx[sch:]
            sigys = model.module.vone_block.sigy[0:sch]
            sigyc = model.module.vone_block.sigy[sch:]

            sigxys = np.sqrt(sigxs*sigys)
            sigxyc = np.sqrt(sigxc*sigyc)

            nxs = sigxs * sfs
            nxc = sigxc * sfc
            nys = sigys * sfs
            nyc = sigyc * sfc

            ppd = model.module.image_size / model.module.visual_degrees
            sfs = sfs * ppd
            sfc = sfc * ppd

            ## Process orientation
            thetas = model.module.vone_block.theta[0:sch]
            thetac = model.module.vone_block.theta[sch:]
            thetas=thetas/np.pi * 180
            thetac=thetac/np.pi * 180

            # Define threshold, the min number of weights in a bin to be considered relevant
            thresh = 0

            Xs3 = [0.5, 1.4, 4, 11.2]
            Xc3 = [0.5, 1.4, 4, 11.2]       
            Yxs = np.logspace(-1, 0., 5, base=10)
            Yxc = np.logspace(-1, 0., 5, base=10)

            records.append(make_colormap_no_and_mean(wbs_means,wbc_means, sfs,nxs, Xs3,Yxs, sfc,nxc, Xc3,Yxc, thresh, 
                          'sf, log2', 'nx, log10', 'Colormap: Weights | Up: no of neurons. Down: means', '_weights_sf3_nx_no_mean_colormap.png'))
            
        
            

        if FLAGS.plot_activ_VOne == True:
            v1_model = vonenet.get_model(model_arch=None, pretrained=False, image_size=64, 
                      visual_degrees=FLAGS.visual_degrees, stride=FLAGS.stride, ksize=FLAGS.ksize, diff_n=FLAGS.diff_n,
                      sf_corr=FLAGS.sf_corr, sf_max=FLAGS.sf_max, sf_min=FLAGS.sf_min, rand_param=FLAGS.rand_param,
                      gabor_seed=FLAGS.gabor_seed, simple_channels=FLAGS.simple_channels,
                      complex_channels=FLAGS.simple_channels, noise_mode=FLAGS.noise_mode,
                      noise_scale=FLAGS.noise_scale, noise_level=FLAGS.noise_level, k_exc=FLAGS.k_exc).module
            
            data_path = '../tiny-imagenet-200/val'
            bsize = 1000

            dataset = torchvision.datasets.ImageFolder(data_path,
                            torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(mean=norm_mean, std=norm_std),
                        ]))

            data_loader = torch.utils.data.DataLoader(dataset, batch_size=bsize, shuffle=True, num_workers=8, pin_memory=True)

            dataloader_iterator = iter(data_loader)
            X, _ = next(dataloader_iterator)

            ## Get activations of the VOneBlock for the batch 
            activations = v1_model(X)
            print(activations.shape)        # should be (n_batch x allch x 32 x 32)

            mean_activations = torch.mean(activations, (0,2,3))   # average over all images in the batch
            print('Mean activations on this batch, one value per channel')

            stddev_activ_i = torch.std(activations, dim=0)
            mean_activ_i = torch.mean(activations, (0))
            cv_activ_i = stddev_activ_i / mean_activ_i

            stddev_activ = torch.mean(stddev_activ_i, (1,2))
            mean_activ = torch.mean(mean_activ_i, (1,2))
            cv_activ = torch.mean(cv_activ_i, (1,2))
            
            sparsenessinterm = np.zeros((allch,32,32))
            for j in range(32):
                for k in range(32):
                    for i in range(allch):
                        sparsenessinterm[i,j,k] = (1 - (torch.sum(activations[:,i,j,k])/bsize).pow(2) / 
                                        torch.sum(activations[:,i,j,k].pow(2)/bsize)) / (1-1/bsize)
            sparseness = torch.mean(torch.from_numpy(sparsenessinterm), (1,2))


            ##########################################################################################

            activs = mean_activations[:sch]
            activc = mean_activations[sch:]
            
            sfs = model.module.vone_block.sf[:sch]
            sfc = model.module.vone_block.sf[sch:]

            nxs = sigxs * sfs
            nxc = sigxc * sfc
            nys = sigys * sfs
            nyc = sigyc * sfc
            
            ppd = model.module.image_size / model.module.visual_degrees
            sfs = sfs * ppd
            sfc = sfc * ppd

            sigxs = model.module.vone_block.sigx[:sch]
            sigxc = model.module.vone_block.sigx[sch:]
            sigys = model.module.vone_block.sigy[:sch]
            sigyc = model.module.vone_block.sigy[sch:]

            thetas = model.module.vone_block.theta[:sch]
            thetac = model.module.vone_block.theta[sch:]


            ########################
            ## ACTIVATIONS colormaps
            ########################

            # Define threshold, the min number of activations in a bin to be considered relevant
            thresh = 0
            
            records.append(make_colormap_no_and_mean(activs,activc, sfs,nxs, Xs3,Yxs, sfc,nxc, Xc3,Yxc, thresh, 
                          'sf, log2', 'nx, log10', 'Colormap: Activations neurons and mean | Ox: sf, Oy: nx', '_activations_mean_nx_colormap.png'))
            records.append(make_colormap_no_and_mean(cv_activ[:sch],cv_activ[sch:], sfs,nxs, Xs3,Yxs, sfc,nxc, Xc3,Yxc, thresh, 
                          'sf, log2', 'nx, log10', 'Colormap: Activations coeff of var | Ox: sf, Oy: nx', '_activations_cv_nx_colormap.png'))
            records.append(make_colormap_no_and_mean(sparseness[:sch],sparseness[sch:], sfs,nxs, Xs3,Yxs, sfc,nxc, Xc3,Yxc, thresh, 
                          'sf, log2', 'nx, log10', 'Colormap: Activations sparseness | Ox: sf, Oy: nx', '_activations_spars_nx_colormap.png'))
            
            

            ########################################################

            results = {'gabor_seed': str(FLAGS.gabor_seed), 
                        'cell_type': 'all_cells', 
                        'sfs': sfs,
                        'sfc': sfc,
                        'nxs': nxs,
                        'nxc': nxc,
                        'nys': nys,
                        'nyc': nyc,
                        'thetas': thetas,
                        'thetac': thetac,
                        'ws': wbs_means,
                        'wc': wbc_means,
                        'as': activs,
                        'ac': activc, 
                        'acvs': cv_activ[:sch], 
                        'acvc': cv_activ[sch:],
                        'asps': sparseness[:sch],
                        'aspc': sparseness[sch:],
                        'astdevs': stddev_activ[:sch],
                        'astdevc': stddev_activ[sch:]
                    }
            records.append(results)


    tester = TinyImageNetTest(model)

    recent_time = time.time()
    results = {'meta': {'phase': 'test',
                        'run': FLAGS.model_path,
                        'wall_time': recent_time}
               }

    if FLAGS.output_path is not None:
        if not (os.path.isdir(FLAGS.output_path)):
            os.mkdir(FLAGS.output_path)

    results[tester.name] = tester()

    if len(results) > 1:
        pprint.pprint(results)
        records.append(results)

    if len(records) > 1:
        pickle.dump(records, open(os.path.join(FLAGS.output_path, FLAGS.results_postproc_filename), 'wb'))




class TinyImageNetTest(object):

    def __init__(self, model):
        self.name = 'test'
        self.model = model
        self.data_loader = self.data()
        self.loss = nn.CrossEntropyLoss(size_average=False)
        self.loss = self.loss.to(device)

    def data(self):
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(FLAGS.in_path, 'val'),
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
        record['gabor_seed'] = str(FLAGS.gabor_seed)

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
