# M-FAC

This repository contains efficient reference implementations of the static and
dynamic M-FAC algorithms, introduced in the paper *"M-FAC: Efficient Matrix-Free 
Approximations of Second-Order Information"* to be published at NeurIPS 2021, plus 
some sample code demonstrating their use in optimization and pruning. 

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
- An implementation of a pruner that utilizes the static algorithm:
  `main_prun.py`
- A script for running simple gradual and one-shot (also with recomputation)
  pruning experiments: `main_prun.py`
- Some standard library code for models, data loading, etc.

## Optimization:

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

## Pruning:

The file `main_prun.py` provides a simple interface for executing various
gradual and one-shot experiments.  Only support for ResNet20/CIFAR with a
corresponding pretrained model is currently included, but other models should
be straight-forward to add. Here follows an example call: 

```
CUDA_VISIBLE_DEVICES=0 python3 -u main_prun.py \
	--model resnet20 \
	--checkpoint checkpoints/resnet20_cifar10.pth.tar \
	--data datasets \
	--nepochs 10 \
	--optim sgd \
	--lr .005 \
	--momentum .9 \
	--batchsize 128 \
	--drop_at 7 9 \
	--pruner mfac \
	--blocksize 128 \
	--nrecomps 16 \
	--ngrads_schedule 64 \
	--sparsities .5 .75 .875 \
	--prun_every 2 \
	--prun_lrs .005 .0005 \
	--prefix experiments/rn20-test/model
```

Most arguments are straight-forward, see also `--help` for their descriptions.
For gradual pruning, the key arguments are `--sparsities`, `--prun_every` and
`--prun_lrs`. The first specifies the individual pruning steps in terms of
overall sparsity relative to all *pruned* parameters (`--adjust_sparsities`
automatically turns these into overall sparsities with respect to *all*
parameters), starting with initial pruning before epoch
0. The second defines how many finetuning epochs there are in between pruning
   steps while the third gives the learning rates to use for those (as dicussed
in the paper, we find that dropping the learning rate one epoch before the next
pruning step can be helpful). After the last pruning step is complete,
additional finetuning will begin with base learning rate `--lr` which is
dropped by `--drop_by` (default 0.1) at epochs `--drop_at` (overall, i.e. also
counting the gradual pruning ones). For oneshot experiments, simply set
`--nepochs` to 0.

There are also additional paramters `--blocksize` for blocked estimation (with
the advanced optimization parameter `--perbatch` to specify how many blocks are
to be handled simultaneously) and `--pages` to specify how many pages to use
for a full blocksize estimation where the gradients do not fully fit into GPU
memory (used when `--blocksize` is -1).

## Models:

Checkpoints of the following sparse MobileNetV1 and ResNet50 models from the 
practical pruning experiments can be found 
[at this link](https://seafile.ist.ac.at/d/5d06074221604d90909b/). They are 
compatible with the model definitions in the 
[STR repository](https://github.com/RAIVNLab/STR).

| Model @ Sparsity | MBv1 @ 75% | MBv1 @ 89% | RN50 @ 95% | RN50 @ 98% |
| ---------------- | :--------: | :--------: | :--------: | :--------: |
| Accuracy         | 70.9       | 67.2       | 72.6       | 67.6       |

 Furthermore, our best finetuned BERT-tiny and BERT-mini models for SQuADv2 and 
 GLUE tasks are uploaded to [HuggingFace Hub](https://huggingface.co/M-FAC).

## BibTeX

To be updated after NeurIPS 2021.

```
@article{DBLP:journals/corr/abs-2107-03356,
  author    = {Elias Frantar and
               Eldar Kurtic and
               Dan Alistarh},
  title     = {Efficient Matrix-Free Approximations of Second-Order Information,
               with Applications to Pruning and Optimization},
  journal   = {CoRR},
  volume    = {abs/2107.03356},
  year      = {2021},
  url       = {https://arxiv.org/abs/2107.03356},
  eprinttype = {arXiv},
  eprint    = {2107.03356},
  timestamp = {Tue, 20 Jul 2021 15:08:33 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2107-03356.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
