# Script for running simple optimizer experiments.


import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib.data.datasets import *
from optim import *


N_GRADS = 1024
N_PERGRAD = 128

EVAL_BATCHSIZE = N_PERGRAD
N_WORKERS = 6


# Test model `model` on dataset `data` using batchsize `batch_size`
@torch.no_grad()
def test(model, data, batch_size=EVAL_BATCHSIZE):
    preds = []
    ys = []
    for batch in DataLoader(data, shuffle=True, batch_size=batch_size, num_workers=N_WORKERS, pin_memory=True):
        x, y = batch
        x = x.to(model.device)
        y = y.to(model.device)
        preds.append(torch.argmax(model(x), 1))
        ys.append(y)
    return torch.mean((torch.cat(preds) == torch.cat(ys)).float()).item()


# Train model `model` on dataset `train_data` using optimizer `optim`, batchsize `batch_size` and 
# decaying the learning rate by a factor of 0.1 before each epoch in `decay_at`. Further, test the
# current model after each epoch on dataset `test_data` and save a checkpoint as file `save`.
def train(
    model, train_data, optim, nepochs,
    test_data=None, decay_at=[], batch_size=N_PERGRAD, save='trained.pth'
):
    criterion = nn.functional.cross_entropy
    for i in range(nepochs):
        tick = time.time()
        print(i)
        if i in decay_at:
            for param_group in optim.param_groups:
                param_group['lr'] *= .1
        runloss = 0.
        step = 0
        for x, y in DataLoader(
            train_data, shuffle=True, batch_size=batch_size, num_workers=N_WORKERS, pin_memory=True
        ):
            x = x.to(model.device)
            y = y.to(model.device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            runloss += loss.item()
            loss.backward()
            optim.step()
            step += 1
        if test_data is not None:
            model.eval()
            print('test: %.4f' % test(model, test_data))
            model.train()
        print('loss: %.4f' % (runloss / step))
        print('time: %.1f' % (time.time() - tick))
        torch.save(model.state_dict(), save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=['resnet20', 'resnet32', 'mobilenet', 'wideresnet-22-2', 'wideresnet-40-2', 'wideresnet-22-4'], required=True,
        help='Type of model to train.'
    )
    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='Path to dataset to use for training.'
    )
    parser.add_argument(
        '--optim', choices=['sgd', 'adam', 'mfac'], required=True,
        help='Type of optimizer to use for training.'
    )
    parser.add_argument(
        '--ngrads', type=int, default=N_GRADS,
        help='Size of the gradient buffer to use for the M-FAC optimizer.'
    )
    parser.add_argument(
        '--save', type=str, default='trained.pth',
        help='Name of the file where the checkpoint of the most recent epoch is persisted.'
    )
    parser.add_argument(
        '--batchsize', type=int, default=N_PERGRAD,
        help='Batchsize to use for training.'
    )
    parser.add_argument(
        '--momentum', type=float, default=0,
        help='Momentum to use for the optimizer.'
    )
    parser.add_argument(
        '--weightdecay', type=float, default=0,
        help='Weight decay to use for the optimizer.'
    )
    args1 = parser.parse_args()

    if args1.model == 'resnet20':
        from lib.models.resnet_cifar10 import *
        model = resnet20()
        train_data, test_data = get_datasets('cifar10', args1.dataset_path)
    if args1.model == 'resnet32':
        from lib.models.resnet_cifar10 import *
        model = resnet32()
        train_data, test_data = get_datasets('cifar10', args1.dataset_path)
    if args1.model == 'mobilenet':
        from lib.models.mobilenet import *
        model = mobilenet()
        train_data, test_data = get_datasets('imagenet', args1.dataset_path)
    if args1.model == 'wideresnet-22-2':
        from lib.models.wide_resnet import *
        model = Wide_ResNet(22, 2, 0, 100)
        train_data, test_data = get_datasets('cifar100', args1.dataset_path)
    if args1.model == 'wideresnet-40-2':
        from lib.models.wide_resnet import *
        model = Wide_ResNet(40, 2, 0, 100)
        train_data, test_data = get_datasets('cifar100', args1.dataset_path)
    if args1.model == 'wideresnet-22-4':
        from lib.models.wide_resnet import *
        model = Wide_ResNet(22, 4, 0, 100)
        train_data, test_data = get_datasets('cifar100', args1.dataset_path)

    dev = torch.device('cuda:0')
    torch.cuda.set_device(dev)
    model = model.to(dev)
    model.device = dev
    model.dtype = torch.float32
    model.train()

    if torch.cuda.device_count() == 1:
        gpus = [dev]
    else:
        gpus = [torch.device('cuda:' + str(i)) for i in range(1, torch.cuda.device_count())]

    optim = {
        'sgd': lambda: torch.optim.SGD(
            model.parameters(),
            lr=.1, momentum=.9, weight_decay=args1.weightdecay
        ),
        'adam': lambda: torch.optim.Adam(model.parameters()),
        'mfac': lambda: MFAC(
            model.parameters(), ngrads=args1.ngrads, lr=.001, damp=1e-5,
            moddev=dev, optdev=dev,
            gpus=gpus,
            momentum=args1.momentum, weight_decay=args1.weightdecay
        ),
    }[args1.optim]()

    print('Training ...')

    if args1.model in ['resnet20', 'resnet32']:
        train(
            model, train_data, optim, 164, decay_at=[82, 123],
            batch_size=args1.batchsize, test_data=test_data, save=args1.save
        )
    if 'wideresnet' in args1.model:
        train(
            model, train_data, optim, 200, decay_at=[100, 150],
            batch_size=args1.batchsize, test_data=test_data, save=args1.save
        )
    if args1.model in ['mobilenet']:
        train(
            model, train_data, optim, 100, decay_at=[50, 75],
            batch_size=args1.batchsize, test_data=test_data, save=args1.save
        )
