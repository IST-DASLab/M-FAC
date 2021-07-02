# Contains a full implementation of the dynamic algorithm and the M-FAC optimizer


import torch
import torch.nn as nn


# Disable tensor cores as they can mess with precision
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# Use custom CUDA implementation for computing $B$ / `coef` if installed 
USE_CUDA = True 
try:
    import hinv_cuda
except Exception as e:
    USE_CUDA = False


# Full implementation of the dynamic algorithm with support for splitting gradients across GPUs
class HInvFastUpMulti:

    # `dev` is the device where all the coefficient calculation happens
    # `grads` ... initial $m \times d$ gradient matrix $G$; assumed to be on CPU (can be deleted afterwards)
    # `dev` ... device where coefficient calculation happens
    # `gpus` ... list of GPUs across which to split stored gradients 
    # `damp` ... dampening constant $\lambda$
    def __init__(self, grads, dev, gpus, damp=1e-5):
        self.m, self.d = grads.shape
        self.dev = dev 
        self.gpus = gpus
        self.dtype = grads.dtype
        self.lambd = 1. / damp

        if USE_CUDA and self.m % 32 != 0 or self.m > 1024:
            raise ValueError('CUDA implementation currently on supports $m$ < 1024 and divisible by 32.')

        self.dper = self.d // len(gpus) + 1 
        self.grads = [] # matrix $G$ in the paper
        for i in range(len(gpus)):
            start, end = i * self.dper, (i + 1) * self.dper
            self.grads.append(grads[:, start:end].to(gpus[i]))
        self.dots = torch.zeros((self.m, self.m), device=self.dev, dtype=self.dtype) # matrix $GG^T$
        for i in range(len(gpus)):
            self.dots += self.grads[i].matmul(self.grads[i].t()).to(self.dev)

        self.last = 0 # ringbuffer index
        self.giHig = self.lambd * self.dots # matrix $D$
        self.denom = torch.zeros(self.m, device=self.dev, dtype=self.dtype) # $D_ii + m$ 
        self.coef = self.lambd * torch.eye(self.m, device=self.dev, dtype=self.dtype) # matrix $B$
        self.setup()

    # Calculate $D$ / `giHig` and $B$ / `coef`
    def setup(self):
        self.giHig = self.lambd * self.dots
        diag = torch.diag(torch.full([self.m], self.m, device=self.dev, dtype=self.dtype))
        self.giHig = torch.lu(self.giHig + diag, pivot=False)[0]
        self.giHig = torch.triu(self.giHig - diag)
        self.denom = self.m + torch.diagonal(self.giHig)
        tmp = -self.giHig.t().contiguous() / self.denom.reshape((1, -1))

        if USE_CUDA:
            diag = torch.diag(torch.full([self.m], self.lambd, device=self.dev, dtype=self.dtype))
            self.coef = hinv_cuda.hinv_setup(tmp, diag)
        else:
            for i in range(max(self.last, 1), self.m):
                self.coef[i, :i] = tmp[i, :i].matmul(self.coef[:i, :i])

    # Distributed `grads.matmul(x)`
    def grads_matmul(self, x):
        def f(i):
            start, end = i * self.dper, (i + 1) * self.dper
            G = self.grads[i]
            return G.matmul(x[start:end].to(G.device)).to(self.dev)
        outputs = nn.parallel.parallel_apply(
            [f] * len(self.gpus), list(range(len(self.gpus)))
        )
        return sum(outputs)

    # Distributed `x.matmul(grads)`
    def matmul_grads(self, x):
        def f(G):
            return (x.to(G.device).matmul(G)).to(self.dev)
        outputs = nn.parallel.parallel_apply(
            [f] * len(self.grads), self.grads
        )
        return torch.cat(outputs)

    # Distributed `grads[j, :] = g`
    def set_grad(self, j, g):
        def f(i):
            start, end = i * self.dper, (i + 1) * self.dper
            self.grads[i][j, :] = g[start:end]
        nn.parallel.parallel_apply(
            [f] * len(self.grads), list(range(len(self.gpus)))
        )

    # Product with inverse of dampened empirical Fisher
    def mul(self, x, dots=None):
        if dots is None:
            dots = self.grads_matmul(x)
        giHix = self.lambd * dots 
        if USE_CUDA:
            giHix = hinv_cuda.hinv_mul(self.giHig, giHix)
        else:
            for i in range(1, self.m):
                giHix[i:].sub_(self.giHig[i - 1, i:], alpha=giHix[i - 1] / self.denom[i - 1])
        return self.lambd * x - self.matmul_grads((giHix / self.denom).matmul(self.coef))

    # Replace oldest gradient with `g` 
    def update(self, g):
        self.set_grad(self.last, g)
        tmp = self.grads_matmul(g)
        self.dots[self.last, :] = tmp
        self.dots[:, self.last] = tmp
        self.setup()
        self.last = (self.last + 1) % self.m

    # Replace oldest gradient with `g` and then calculate the IHVP with `g`
    def update_mul(self, g):
        self.set_grad(self.last, g)
        tmp = self.grads_matmul(g)
        self.dots[self.last, :] = tmp
        self.dots[:, self.last] = tmp
        self.setup()
        res = self.mul(g, tmp)
        self.last = (self.last + 1) % self.m
        return res


# PyTorch compatible implementation of the M-FAC optimizer 
class MFAC(torch.optim.Optimizer):

    # `params` ... model parameters to optimize 
    # `lr` ... learning rate
    # `momentum` ... momentum coefficient
    # `weight_decay` ... weight decay constant
    # `ngrads` ... size of gradient window 
    # `damp` ... dampening constant $\lambda$
    # `moddev` ... device where the model to be optimized resides 
    # `optdev` ... device where coefficient calculation of the dynamic algorithm happens
    # `gpus` ... list of GPUs where the stored gradients are distributed to
    # `sparse` ... whether do to sparse optimization; weights initially exactly 0 will be considered as dropped
    def __init__(
        self, params, 
        lr=1e-3, momentum=0, weight_decay=0, 
        ngrads=1024, damp=1e-5,
        moddev=None, optdev=None, gpus=[], 
        sparse=False
    ):
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.moddev = moddev
        self.optdev = optdev
        self.sparse = sparse

        super(MFAC, self).__init__(params, dict(lr=lr))

        with torch.no_grad():
            w = []
            for group in self.param_groups:
                for p in group['params']:
                    w.append(p.reshape(-1))
            w = torch.cat(w)
            w = w.to(self.optdev)
            self.nweights = w.numel()
            if self.sparse:
                self.mask = w != 0
                w = w[self.mask]

            if self.momentum > 0:
                self.v = torch.zeros(w.shape, device=self.optdev)
            if len(gpus) == 0:
                gpus = [self.optdev]
            self.hinv = HInvFastUpMulti(
                torch.zeros((ngrads, w.numel()), dtype=torch.float), dev=self.optdev, gpus=gpus, damp=damp
            )

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('`closure` not supported')

        g = []
        for group in self.param_groups:
            for p in group['params']:
                if self.weight_decay > 0:
                    tmp = p.grad + self.weight_decay * p
                else:
                    tmp = p.grad
                g.append(tmp.reshape(-1))
        g = torch.cat(g)
        g = g.to(self.optdev)
        if self.sparse:
            g = g[self.mask]

        tmp = self.hinv.update_mul(g)
        if self.momentum > 0:
            self.v = self.momentum * self.v + (1 - self.momentum) * tmp
            tmp = self.v
        if self.sparse:
            expanded = torch.zeros(self.nweights, device=self.optdev)
            expanded[self.mask] = tmp
            tmp = expanded
        tmp = tmp.to(self.moddev)

        count = 0
        for group in self.param_groups:
            for p in group['params']:
                p.add_(tmp[count:(count + p.numel())].reshape(p.shape), alpha=-group['lr'])
                count += p.numel()


# Naive Woodbury implementation for testing correctness 
class HInvSlow:
    
    def __init__(self, grads, damp=1e-5):
        m, d = grads.shape
        H = torch.diag(torch.full([d], 1. / damp, dtype=grads.dtype, device=grads.device))
        for i in range(m):
            g = grads[i, :]
            Hg = H.matmul(g)
            H -= torch.ger(Hg, Hg) / (m + g.dot(Hg))
        self.H = H

    def mul(self, x):
        return self.H.matmul(x)


# Small test comparing dynamic algorithm results with naive Woodbury implementation
if __name__ == '__main__':
    D = 1000 
    M = 32 
    DEV = torch.device('cuda:0')

    def dist(x, y):
        return torch.mean(torch.abs(x - y))

    grads = torch.randn((M, D), device=DEV, dtype=torch.float64)
    g = torch.randn(D, device=DEV, dtype=torch.float64)
    hinv1 = HInvFastUpMulti(torch.zeros((M, D), dtype=torch.float64), DEV, [DEV])
    hinv2 = HInvSlow(grads.clone())

    for i in range(M):
        hinv1.update(grads[i, :])
    print(dist(hinv1.mul(g), hinv2.mul(g)))
