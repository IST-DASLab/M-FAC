# Script for running simple oneshot pruning experiments.


import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from prun import *
from lib.data.datasets import *


N_GRADS = 1024 
N_PERGRAD = 32 

EVAL_BATCHSIZE = 256 
N_WORKERS = 6 


def load_model(path, model):
    tmp = torch.load(path)['state_dict']
    for k in list(tmp.keys()):
        tmp[k.replace('module.', '')] = tmp[k]
        del tmp[k]
    model.load_state_dict(tmp)


@torch.no_grad()
def test(model, data, batch_size=EVAL_BATCHSIZE):
    preds = []
    ys = []
    for batch in DataLoader(
        data, shuffle=True, batch_size=batch_size, num_workers=N_WORKERS, pin_memory=True
    ):
        x, y = batch
        x = x.to(model.device)
        y = y.to(model.device)
        preds.append(torch.argmax(model(x), 1))
        ys.append(y)
    return torch.mean((torch.cat(preds) == torch.cat(ys)).float()).item()

@torch.no_grad()
def reset_bnstats(model, data, n_batches=1000, batch_size=EVAL_BATCHSIZE):
    i = 0
    while True:
        for batch in DataLoader(
            data, shuffle=True, batch_size=EVAL_BATCHSIZE, num_workers=N_WORKERS, pin_memory=True
        ):
            if i == n_batches:
                return
            x, y = batch
            x = x.to(model.device)
            model(x)
            i += 1


@torch.no_grad()
def get_gvec(model, params):
    named_parameters = dict(model.named_parameters())
    return torch.cat([
        named_parameters[p].grad.reshape(-1) for p in params
    ])

@torch.no_grad()
def zero_grads(model):
    for p in model.parameters():
        p.grad = None

def collect_grads(model, pruner, data, npergrad=N_PERGRAD):
    named_parameters = dict(model.named_parameters())

    criterion = nn.functional.cross_entropy
    i = 0
    while True:
        for batch in DataLoader(
            data, shuffle=False, batch_size=npergrad, num_workers=N_WORKERS, pin_memory=True
        ):
            if i == pruner.ngrads:
                return 
            x, y = batch
            x = x.to(model.device)
            y = y.to(model.device)

            loss = criterion(model(x), y)
            loss.backward()

            pruner.report_grad(get_gvec(model, pruner.params))
            zero_grads(model)
            i += 1


def oneshot_prune(
    model, pruner, sparsities, recomps, tests, train_data, test_data, npergrad=N_PERGRAD
):
    w = None
    drop = None
    ignore = None
    for i, sparsity in enumerate(sparsities):
        if i in recomps:
            if i > 0:
                pruner.update(w, drop, ignore)
            if pruner.ngrads:
                print('Collecting grads ...')
                collect_grads(model, pruner, train_data)
                zero_grads(model)
            print('Preparing to prune ...')
            pruner.prepare()
        w, drop, ignore = pruner.prune(sparsity - pruner.sparsity())
        if i in tests:
            state_dict = model.state_dict()
            pruner.set_pvec(w)
            print('Evaluating ...')
            print('%.2f %.3f' % (sparsity, test(model, test_data)))
            torch.save(model.state_dict(), 'pruned.pth')
            model.load_state_dict(state_dict)
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', choices=['resnet20'], required=True) 
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--pruner', choices=['mag', 'mfac'], required=True)

    parser.add_argument('--ngrads', type=int, default=N_GRADS)
    parser.add_argument('--blocksize', type=int, default=-1)
    parser.add_argument('--pages', type=int, default=1)
    parser.add_argument('--perbatch', type=int, default=1)
    parser.add_argument('--npergrad', type=int, default=N_PERGRAD)

    parser.add_argument('--sparsities', type=float, nargs='+', default=[1.])
    parser.add_argument('--recomps', type=int, nargs='+', default=[0])
    parser.add_argument('--tests', type=int, nargs='+', default=[0])

    args1 = parser.parse_args()

    if args1.model == 'resnet20':
        from lib.models.resnet_cifar10 import *
        model = resnet20()
        load_model('checkpoints/resnet20_cifar10.pth.tar', model)
        train_data, test_data = cifar10_get_datasets(args1.dataset)
        params = [
            n for n, _ in model.named_parameters() if ('conv' in n or 'fc' in n) and 'bias' not in n
        ]

    dev = torch.device('cuda:0')
    model = model.to(dev)
    model.device = dev
    model.dtype = torch.float32
    
    if args1.model == 'resnet20':
        # Fix messed up batch-norm stats of pretrained model
        reset_bnstats(model, train_data)
    model.eval() # freeze batch-norm params

    pruners = {
        'mag': lambda: MagPruner(model, params), 
        'mfac': lambda: MFACPruner(
            model, params, args1.ngrads, 
            blocksize=args1.blocksize, pages=args1.pages, perbatch=args1.perbatch
        )
    }
    pruner = pruners[args1.pruner]()

    oneshot_prune(
        model, pruner,
        args1.sparsities, args1.recomps, args1.tests, 
        train_data, test_data, 
        npergrad=args1.npergrad
    )
