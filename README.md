
This is the code accompanying the paper 

## "Contribution of V1 receptive field properties to corruption robustness in CNNs", by Ruxandra Barbulescu, Tiago Marques and Arlindo Oliveira, ECAI 2024

We create four variants of the VOneResNet18 model by changing how the RF properties were sampled to instantiate the VOneBlock.
- [BIOL]: sampling neuronal RF properties from biological empirical distributions obtained from published studies;
- [UNIF1]: sampling uniform, independent distributions within the same range as the empirical distributions;
- [UNIF2]: like the previous one but introducing a correlation between nx and ny to restrict the range of RF aspect ratios to one that is closer to biology;
- [UNIF3]: like the previous one but introducing a correlation between nx and spatial frequency (SF) to avoid combinations of values not found in the primate V1 (very large SF and low nx and vice-versa).

The VOneNets are a family of biologically-inspired Convolutional Neural Networks (CNNs). VOneNets have the following features:
- Fixed-weight neural network model of the primate primary visual cortex (V1) as the front-end.
- Robust to image perturbations
- Brain-mapped
- Flexible: can be adapted to different back-end architectures


## Requirements

- Python 3.6+
- PyTorch 1.10.1+
- numpy
- pandas
- tqdm
- scipy

- requests
- matplotlib
- torchvision
- fire (for train)


## Citation

Barbulescu, R., Marques, T., Oliveira, A., "Contribution of V1 receptive field properties to corruption robustness in CNNs", ECAI 2024


## Setup and Run

1. You need to clone it in your local repository
  $ git clone https://github.com/ruxandrab/vonenet-rf.git
   
2. Get the Tiny ImageNet dataset from:
  https://www.kaggle.com/datasets/zhanghongyu111/tiny-imagenet-200

3. Example of running a test for a pretrained model:
  $ test.py --in_path='../tiny-imagenet-200' --model_path='model/Run1' --model_arch='voneresnet18' --gabor_seed=0 --k_exc=23  --show_or_save_figs='save' test 

4. Example of training a model from the scratch:
  $ python train.py --in_path='../tiny-imagenet-200' --output_path='./model/Run1' --model_arch='voneresnet18' --gabor_seed=0 --simple_channels=256 --complex_channels=256 --k_exc=23 train
  
  
