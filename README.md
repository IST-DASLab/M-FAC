This repository contains efficient reference implementations of the static and
dynamic M-FAC algorithms plus some sample code demonstrating their use in
optimization and pruning. 

More concretely, it contains the following:

- An efficient full implementation of the dynamic algorithm including custom
  CUDA kernels: `optim.py`, `hinv_cuda_kernel.cu`, `hinv_cuda.cpp`,
`setup_cuda.py`
- A PyTorch compatible implementation of the M-FAC optimizer: `optim.py`
- A script for running simple optimization experiments: `main_optim.py`
- An efficient implementation of the static algorithm in blocked form with
  simultaneous handling of multiple blocks: `prun.py`
- An implementation of the full (non-blocked) static algorithm with efficient
  paging: `prun.py` 
- An implementation of a pruner that utilizes the static algorithm: `prun.py`
- A script for running simple one-shot (& recomputation) experiments: `main_prun.py`
- Some standard library code for models, data loading, etc.

Optimization:
=============

The CUDA kernels for more efficient coefficient computation: `python
setup_cuda.py install`. The code runs without doing so, but substantially
slower.

The file `main_optim.py` provides a simple command-line interface for running
dense M-FAC optimization experients. A sample call could look like this:

```
CUDA_VISIBLE_DEVICES=0 python3 main_optim.py \
    --model resnet20 \
    --dataset DATASET \
    --optim mfac \
    --ngrads 512 \
    --weightdecay .003 \
    --batchsize 128 \
    --save rn20-mfac.pth \
> rn20-mfac.txt
```

`--model` specifies the model to optimize (see `--help` for a full list of
model names), `--dataset` the path to the dataset (for CIFAR models, the data
is automatically downloaded if it does not yet exist), `--optim` specifies the
optimizer to use, `--ngrads` the size of the sliding window for the dynamic
algorithm, `--weightdecay` the weight decay, `--batchsize` the batch size and
`--save` the name of the file where the model is stored after each epoch.

Finally, it is worth noting that the optimizer implementation `optim.MFAC` also
has support for sparse optimization.

Pruning:
========

The file `main_prun.py` provides a simple interface for executing various
one-shot experiments.  Only support for ResNet20/CIFAR with a corresponding
pretrained model is currently included, but other models should be
straight-forward to add. Here follows an example call: 

```
python3 main_prun.py \
    --model resnet20 \
    --dataset DATASET \
    --pruner mfac \
    --ngrads 1024 \
    --sparsities 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
    --recomps 0 1 2 3 4 5 6 7 8 \
    --tests 0 1 2 3 4 5 6 7 8
```

The most interesting arguments are `--sparsities`, `--recomps` and `--tests`.
The first specifies different sparsity levels to consider; the second declares
after which pruning steps to recompute the inverse Hessian estimation and the
third after which pruning steps to evaluate the model. For example, the call
above indicates that we are interested in pruning our model to 9 different
sparsity levels, while recomputing the inverse Hessian estimation after each
level and always also printing the model accuracy.

There are also some additional paramters `--blocksize` for blocked estimation
(with the advanced optimization parameter `--perbatch` to specify how many
blocks are to be handled simultaneously) and `--pages` to specify how many
pages to use for a full blocksize estimation where the gradients do not fully
fit into GPU memory.
