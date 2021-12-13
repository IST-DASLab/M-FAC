import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from lib.data.datasets import get_datasets 
from prun import *


def load_model(path, model):
    tmp = torch.load(path, map_location='cpu')
    if 'state_dict' in tmp:
        tmp = tmp['state_dict']
    if 'model' in tmp:
        tmp = tmp['model']
    for k in list(tmp.keys()):
        if 'module.' in k:
            tmp[k.replace('module.', '')] = tmp[k]
            del tmp[k]
    model.load_state_dict(tmp)

def load_mask(path):
    tmp = torch.load(path, map_location='cpu')
    return tmp.get('mask', None)

@torch.no_grad()
def test(model, dataloader):
    preds = []
    ys = []
    for batch in dataloader:
        x, y = batch
        x = x.to(model.device)
        y = y.to(model.device)
        preds.append(torch.argmax(model(x), 1))
        ys.append(y)
    return torch.mean((torch.cat(preds) == torch.cat(ys)).float()).item()


@torch.no_grad()
def get_pvec(model, params):
    state_dict = model.state_dict()
    return torch.cat([
        state_dict[p].reshape(-1) for p in params
    ])

@torch.no_grad()
def set_pvec(w, model, params, nhwc=False):
    state_dict = model.state_dict()
    i = 0
    for p in params:
        count = state_dict[p].numel()
        state_dict[p] = w[i:(i + count)].reshape(state_dict[p].shape)
        i += count
    model.load_state_dict(state_dict)

@torch.no_grad()
def get_gvec(model, params):
    named_parameters = dict(model.named_parameters())
    return torch.cat([
        named_parameters[p].grad.reshape(-1) for p in params
    ])

@torch.no_grad()
def apply_mask(mask, model, params):
    state_dict = model.state_dict()
    i = 0
    for p in params:
        param = state_dict[p]
        count = param.numel()
        state_dict[p] *= mask[i:(i + count)].reshape(param.shape).float()
        i += count
    model.load_state_dict(state_dict)
    
@torch.no_grad()
def zero_grads(model):
    for p in model.parameters():
        p.grad = None


class MagPruner:

    def __init__(self, model, params):
        self.model = model
        self.params = params

    def prune(self, mask, sparsity):
        w = get_pvec(self.model, self.params)
        dev = w.device
        ndrop = int(w.numel() * (sparsity - torch.mean((~mask).float())))
        w1 = w[mask] 
        d1 = w1.numel()

        p = torch.abs(w1) 
        psorted = torch.argsort(p)
        drop = psorted[:ndrop]

        w1[drop] = 0
        w[mask] = w1
        tmp = torch.ones_like(w1)
        tmp[drop] = 0
        mask[mask] = tmp > 0
        return mask, w

class MFACPruner:

    def __init__(
        self,
        model, params,
        prunloader, criterion,
        ngrads, blocksize, perbatch, damp, npages 
    ):
        self.model = model
        self.params = params
        self.prunloader = prunloader
        self.criterion = criterion
        self.ngrads = ngrads
        self.blocksize = blocksize
        self.perbatch = perbatch
        self.damp = damp
        self.npages = npages

    def prune(self, mask, sparsity):
        w = get_pvec(self.model, self.params)
        dev = w.device
        ndrop = int(w.numel() * (sparsity - torch.mean((~mask).float())))
        w1 = w[mask] 
        d1 = w1.numel()

        zero_grads(self.model)
        self.model.eval()
        grads = torch.zeros((self.ngrads, d1), device='cpu')
        for i, batch in enumerate(self.prunloader):
            if (i + 1) % self.ngrads == 0:
                break
            x, y = batch
            x = x.to(dev)
            y = y.to(dev)
            loss = self.criterion(self.model(x), y)
            loss.backward()
            grads[i] = get_gvec(self.model, self.params)[mask].to('cpu')
            zero_grads(self.model)
        self.model.train()

        if self.blocksize == -1:
            hinv = HInvFastSwap(grads, self.damp, self.npages)
        else:
            hinv = HInvFastBlock(
                grads, self.blocksize, self.perbatch, dev, damp=self.damp
            )

        diag = hinv.diag().to(dev)
        p = w1 ** 2 / (2. * diag)
        psorted = torch.argsort(p)
        drop = psorted[:ndrop]

        mask1 = mask.clone() 
        tmp1 = torch.ones_like(w1)
        tmp1[torch.argsort(w1 ** 2)[:ndrop]] = 0

        tmp = torch.zeros_like(w1)
        tmp[drop] = -w1[drop] / diag[drop]
        w1 += hinv.mul(tmp).to(dev)
        w1[drop] = 0

        w[mask] = w1
        tmp = torch.ones_like(w1)
        tmp[drop] = 0
        mask[mask] = tmp > 0

        return mask, w

def gradual_prun(
    model, params, mask,
    trainloader, testloader,
    make_optim, nepochs, lr_schedule, 
    pruner, prunepochs, sparsities, ngrads_schedule, nrecomps,
    prefix
):
    runloss = 0.
    step = 0
    optim = make_optim(0, mask)
    sparsities += [torch.mean((~mask).float()).item()]

    oneshot = nepochs == 0 
    if oneshot:
        nepochs = 1
    for epoch in range(nepochs):
        if epoch in prunepochs:
            tick = time.time()
            i = prunepochs.index(epoch)
            if sparsities[i] != 0:
                if isinstance(pruner, MFACPruner):
                    pruner.ngrads = ngrads_schedule[i]
                fac = ((1. - sparsities[i]) / (1. - sparsities[i - 1])) ** (1./nrecomps)
                for j in range(1, nrecomps + 1):
                    mask, w = pruner.prune(mask, 1. - ((1. - sparsities[i - 1]) * (fac**j)))
                    set_pvec(w, model, params)
                duration = time.time() - tick

                model.eval()
                print('prun %02d: sparsity=%.3f, accuracy=%.3f, time=%.1fs' % (
                    i, torch.mean((~mask).float()).item(), test(model, testloader), duration 
                ))
                model.train()
                torch.save(
                    {'model': model.state_dict(), 'mask': mask},
                    '%s-prune-%03d.pth' % (prefix, int(sparsities[i] * 1000))
                )
            else:
                sparsities[0] = sparsities[-1]
            optim = make_optim(epoch, mask)
            if oneshot:
                return

        for param_group in optim.param_groups:
            param_group['lr'] = lr_schedule[epoch]

        tick = time.time()
        for x, y in trainloader:
            x = x.to(dev)
            y = y.to(dev)
            optim.zero_grad()
            loss = criterion(model(x), y)
            runloss = .99 * runloss + .01 * loss.item()
            loss.backward()
            optim.step()
            apply_mask(mask, model, params)

            step += 1
            if step % 100 == 0:
                print('step %06d: runloss=%.3f' % (step, runloss))
        duration = time.time() - tick

        model.eval()
        print('epoch %02d: accuracy=%.3f, time=%.1fs' % (epoch, test(model, testloader), duration))
        model.train()
        torch.save(
            {'model': model.state_dict(), 'mask': mask, 'epoch': epoch}, '%s-last.pth' % prefix
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', choices=['resnet20'], required=True,
        help='Model type.'
    )
    parser.add_argument(
        '--checkpoint', type=str, default='',
        help='Checkpoint to start pruning from.'
    )
    parser.add_argument(
        '--data', type=str, required=True,
        help='Path to dataset.'
    )
    parser.add_argument(
        '--prefix', type=str, default='experiment/model',
        help='Prefix for intermediate checkpoints and results.' 
    )
    parser.add_argument(
        '--adjust-sparsities', action='store_true',
        help='Adjust sparsities relative from relative to pruned layers to global.'
    )

    parser.add_argument(
        '--batchsize', type=int, default=128,
        help='Dataloader (and optimizer) batch size.'
    )
    parser.add_argument(
        '--nepochs', type=int, required=True,
        help='Total number of epochs.'
    )
    parser.add_argument(
        '--nworkers', type=int, default=8,
        help='Number of dataloader workers.'
    )

    parser.add_argument(
        '--optim', choices=['sgd'], default='sgd',
        help='Optimizer to use.'
    )
    parser.add_argument(
        '--lr', type=float, default=.005,
        help='Optimizer base learning rate.'
    )
    parser.add_argument(
        '--momentum', type=float, default=.9,
        help='Optimizer momentum.'
    )
    parser.add_argument(
        '--weightdecay', type=float, default=0,
        help='Optimizer weight decay.'
    )
    parser.add_argument(
        '--drop_at', type=int, nargs='+', default=[],
        help='List of epochs where the learning rate is dropped.'
    )
    parser.add_argument(
        '--drop_by', type=float, default=.1,
        help='Factor to drop the learning rate by.'
    )

    parser.add_argument(
        '--pruner', choices=['mag', 'mfac'], default='mag',
        help='Pruner to use.'
    )
    parser.add_argument(
        '--prun_every', type=int, default=1,
        help='Pruner every X epochs.'
    )
    parser.add_argument(
        '--sparsities', type=float, nargs='+', default=[],
        help='List of pruning steps.'
    )
    parser.add_argument(
        '--ngrads_schedule', type=int, nargs='+', default=[64],
        help='Number of gradients to use for M-FAC, either a constant or a list with one value per pruning step.' 
    )
    parser.add_argument(
        '--nrecomps', type=int, default=16,
        help='Number of recomputations to use for M-FAC.'
    )
    parser.add_argument(
        '--prun_batchsize', type=int, default=32,
        help='Batchsize for individual M-FAC gradients.'
    )
    parser.add_argument(
        '--prun_lrs', type=float, nargs='+', default=None,
        help='Cycle through these learning rates in between pruning steps.'
    )
    parser.add_argument(
        '--blocksize', type=int, default=128,
        help='M-FAC blocksize; -1 will use full version.'
    )
    parser.add_argument(
        '--perbatch', type=int, default=10000,
        help='Number of M-FAC blocks to handle simulatenously on the GPU.'
    )
    parser.add_argument(
        '--prundamp', type=float, default=1e-5,
        help='M-FAC dampening.'
    )
    parser.add_argument(
        '--prunpages', type=int, default=1,
        help='Number of pages to use for full M-FAC; i.e. when blocksize is -1.'
    )

    args1 = parser.parse_args()

    if args1.model == 'resnet20':
        from lib.models.resnet_cifar10 import *
        model = resnet20()
        load_model(args1.checkpoint, model)
        train_data, test_data = get_datasets('cifar10', args1.data)
        params = [
            n for n, _ in model.named_parameters() if ('conv' in n or 'fc' in n) and 'bias' not in n
        ]

    dev = torch.device('cuda:0')
    model = model.to(dev)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        params = ['module.' + p for p in params]
    model.device = dev

    trainloader = DataLoader(
        train_data, 
        shuffle=True, batch_size=args1.batchsize, num_workers=args1.nworkers, pin_memory=True
    )
    testloader = DataLoader(
        test_data, 
        shuffle=False, batch_size=args1.batchsize, num_workers=args1.nworkers, pin_memory=True
    )
    prunloader = DataLoader(
        train_data,
        shuffle=True, batch_size=args1.prun_batchsize, num_workers=args1.nworkers, pin_memory=True
    )

    criterion = torch.nn.functional.cross_entropy

    make_pruner = {
        'mag': lambda: MagPruner(model, params),
        'mfac': lambda: MFACPruner(
            model, params, prunloader, criterion, 0, 
            args1.blocksize, args1.perbatch, args1.prundamp, args1.prunpages
        ) 
    }
    pruner = make_pruner[args1.pruner]()

    def make_sgd(epoch, mask):
        return torch.optim.SGD(
            model.parameters(), lr=0, momentum=args1.momentum, weight_decay=args1.weightdecay
        )
    make_optim = {
        'sgd': make_sgd
    }

    prunepochs = [i * args1.prun_every for i in range(len(args1.sparsities))]
    if len(args1.ngrads_schedule) == 1 and len(prunepochs) > 1:
       args1.ngrads_schedule = args1.ngrads_schedule * len(prunepochs)

    if args1.prun_lrs is None:
        args1.prun_lrs = [args1.lr]
    if len(args1.prun_lrs) != args1.prun_every:
        args1.prun_lrs = [args1.prun_lrs[0]] * args1.prun_every
    lr_schedule = args1.prun_lrs * (len(prunepochs) - 1)

    lr = args1.lr
    for epoch in args1.drop_at + [args1.nepochs]:
        lr_schedule += [lr] * (epoch - len(lr_schedule))
        lr *= args1.drop_by

    if args1.adjust_sparsities:
        adjustment = get_pvec(model, [n for n, p in model.named_parameters()]).numel() / get_pvec(model, params).numel()
        args1.sparsities = [s * adjustment for s in args1.sparsities]

    print(prunepochs)
    print(args1.sparsities)
    print(args1.ngrads_schedule)
    print(lr_schedule)

    if '/' in args1.prefix:
        tmp = args1.prefix[:args1.prefix.rfind('/')]
        if not os.path.exists(tmp):
            os.makedirs(tmp)

    # Fix broken batchnorm params for RN20
    if args1.model == 'resnet20':
        model.train()
        for x, y in trainloader:
            x = x.to(dev)
            y = y.to(dev)
            model(x)

    mask = load_mask(args1.checkpoint) if args1.checkpoint != '' else None
    if mask is None:
        mask = torch.ones_like(get_pvec(model, params)) > 0
    mask = mask.to(dev)

    if args1.nepochs > 0:
        model.eval()
        print('initial: sparsity=%.3f, accuracy=%.3f' % (torch.mean((~mask).float()).item(), test(model, testloader)))
        model.train()

    gradual_prun(
        model, params, mask,
        trainloader, testloader,
        make_optim[args1.optim], args1.nepochs, lr_schedule,
        pruner, prunepochs, args1.sparsities, args1.ngrads_schedule, args1.nrecomps,
        args1.prefix
    )
