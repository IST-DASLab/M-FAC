# Contains efficient implementations of the following:
# - Full M-FAC with paging
# - Blocked M-FAC with support for handling multiple blocks at once


import math
import torch
import torch.nn as nn


# Batch matrix/vector operations used for simulatenously precomputing multiple blocks

def batch_dot(A, B):
    b, d = A.shape
    return A.reshape((b, 1, d)).matmul(B.reshape((b, d, 1))).reshape(b)

def batch_mvp(A, B):
    b, d = B.shape
    return A.matmul(B.reshape((b, d, 1))).reshape((b, -1))

def batch_vmp(A, B):
    b, d = A.shape
    return A.reshape((b, 1, d)).matmul(B).reshape((b, -1))


# Full implementation for the static algorithm in blocked form  
class HInvFastBatch:

    # `grads` ... bxmxd matrix of gradients already split into blocks
    # `damp` ... dampening constant $\lambda$
    def __init__(self, grads, damp=1e-5, Hg=None, denom=None):
        if Hg is not None and denom is not None:
            self.dev = Hg.device
            self.dtype = Hg.dtype
            self.b, self.m, self.d = Hg.shape
            self.damp = damp
            self.Hg = Hg
            self.denom = denom
            return

        self.dev = grads.device
        self.dtype = grads.dtype
        b, m, d = grads.shape
        damp = 1. / damp
        Hg = grads 
        denom = torch.zeros((b, m), device=self.dev, dtype=self.dtype)

        g = grads[:, 0, :].clone()
        Hg[:, 0, :] = damp * g 
        denom[:, 0] = m + batch_dot(g, Hg[:, 0, :]) 

        for i in range(1, m):
            g = grads[:, i, :].clone()
            Hg[:, i, :] = damp * g 
            mul = batch_mvp(Hg[:, :i, :], g) / denom[:, :i]
            Hg[:, i, :] -= batch_vmp(mul, Hg[:, :i, :])
            denom[:, i] = m + batch_dot(g, Hg[:, i, :])

        self.b = b
        self.m = m
        self.d = d
        self.damp = damp
        self.Hg = Hg
        self.denom = denom

    # Computes the IHVP with some vector `x`
    def mul(self, x):
        res = self.damp * x
        mul = batch_mvp(self.Hg, x) / self.denom
        res -= batch_vmp(mul, self.Hg)
        return res

    # Returns the diagonal of the inverse Fisher
    def diag(self):
        res = self.damp * torch.ones((self.b, self.d), device=self.dev, dtype=self.dtype)
        for i in range(self.m):
            res -= (self.Hg[:, i, :] ** 2) / self.denom[:, i].reshape((-1, 1))
        return res

    def to(self, dev):
        return HInvFastBatch(None, damp=self.damp, Hg=self.Hg.to(dev), denom=self.denom.to(dev))


# Blocked static algorithm implementation that utilizes CPU memory. 
# It loads and computes with `perbatch` blocks simultaneously, which makes it quite
# efficient, even for smaller blocksizes.
class HInvFastBlock:

    # `grads` ... mxd matrix of gradients; stored in CPU memory
    # `blocksize` ... size of the indvidual blocks 
    # `perbatch` ... number of blocks to handle at the same time on the GPU
    # `gpu` ... GPU device to use for computation
    # `damp` ... dampening constant $\lambda$
    def __init__(self, grads, blocksize, perbatch, gpu, damp=1e-5):
        self.m, self.d = grads.shape
        self.dtype = grads.dtype
        self.blocksize = blocksize
        self.perbatch = perbatch
        self.gpu = gpu

        self.hinvs = []

        k = self.blocksize * self.perbatch
        for i in range(0, self.d, k):
            if i + k > self.d:
                perbatch1 = (self.d - i) // self.blocksize
            else:
                perbatch1 = self.perbatch
            if perbatch1 > 0:
                grads1 = grads[:, i:(i + self.blocksize * perbatch1)].to(gpu)
                grads1 = grads1.t().reshape((perbatch1, -1, self.m))
                grads1 = torch.transpose(grads1, 1, 2).contiguous()
                hinv = HInvFastBatch(grads1, damp=damp)
                self.hinvs.append(hinv.to('cpu'))
                del hinv
            if i + k > self.d:
                i += self.blocksize * perbatch1
                grads1 = grads[:, i:].to(gpu).reshape((1, self.m, -1))
                hinv = HInvFastBatch(grads1, damp=damp)
                self.hinvs.append(hinv.to('cpu'))
                del hinv

        torch.cuda.empty_cache()

    def mul(self, x):
        x = x.to('cpu')
        res = []
        i = 0
        for hinv in self.hinvs:
            hinv = hinv.to(self.gpu)
            k = hinv.b * hinv.d
            tmp = x[i:(i + k)].to(self.gpu).reshape((hinv.b, -1))
            res.append(hinv.mul(tmp).reshape(-1).to('cpu'))
            i += k
            del hinv
        torch.cuda.empty_cache()
        return torch.cat(res)

    def diag(self):
        res = []
        for hinv in self.hinvs:
            hinv = hinv.to(self.gpu)
            res.append(hinv.diag().reshape(-1).to('cpu'))
            del hinv
        torch.cuda.empty_cache()
        return torch.cat(res)


# Implementation of the full static algorithm (no blocking) with intelligent paging 
class HInvFastSwap:
 
    # `grads` ... mxd matrix of gradients; stored in CPU memory;
    #             repurposed to save memory; m must divisible by `npages`
    # `damp` ... dampening constant $\lambda$
    # `npages` ... number of pages to use, i.e. in how many chunks to split `grads`
    # `cpu` ... CPU device to use storage
    # `gpu` ... GPU device to use for computation
    def __init__(self, grads, damp=1e-5, npages=1, cpu='cpu', gpu=torch.device('cuda:0')):
        self.cpu = cpu
        self.gpu = gpu
        
        self.dtype = grads.dtype
        self.m, self.d = grads.shape
        self.damp = 1. / damp
        self.mgpu = self.m // npages

        self.Hg = grads
        self.denom = torch.zeros(self.m, dtype=self.dtype, device=self.gpu)
        self.buf = torch.zeros((self.mgpu, self.d), dtype=self.dtype, device=self.cpu)

        self.comp_ini()
        for off in range(self.mgpu, self.m, self.mgpu):
            self.prep_buf(off)
            self.comp_buf(off)
        del self.buf
        torch.cuda.empty_cache()

    # Precompute the initial block
    def comp_ini(self):
        Hg = self.Hg[:self.mgpu, :].to(self.gpu)

        g = Hg[0, :].clone()
        Hg[0, :] = self.damp * g
        self.denom[0] = self.m + g.dot(Hg[0, :])

        for i in range(1, self.mgpu):
            g = Hg[i, :].clone()
            Hg[i, :] = self.damp * g 
            mul = Hg[:i, :].matmul(g) / self.denom[:i]
            Hg[i, :] -= mul.matmul(Hg[:i, :])
            self.denom[i] = self.m + g.dot(Hg[i, :])

        self.Hg[:self.mgpu, :] = Hg.to(self.cpu)
        del Hg

    # Prepare the buffer
    def prep_buf(self, off):
        for off1 in range(0, off, self.mgpu):
            Hg = self.Hg[off1:(self.mgpu + off1), :].to(self.gpu)

            for j in range(self.mgpu):
                g = self.Hg[j + off, :].to(self.gpu)
                mul = Hg.matmul(g) / self.denom[off1:(self.mgpu + off1)]
                tmp = mul.matmul(Hg)
                if off1 == 0:
                    self.buf[j, :] = (self.damp * g - tmp).to(self.cpu) 
                else:
                    self.buf[j, :] -= tmp.to(self.cpu)
            del Hg

    # Complete precomputation of the current buffer
    def comp_buf(self, off):
        buf = self.buf.to(self.gpu)

        g = self.Hg[off, :].to(self.gpu)
        self.denom[off] = self.m + g.dot(buf[0, :])

        for i in range(1, self.mgpu):
            g = self.Hg[i + off, :].to(self.gpu)
            mul = buf[:i, :].matmul(g) / self.denom[off:(i + off)]
            buf[i, :] -= mul.matmul(buf[:i, :])
            self.denom[i + off] = self.m + g.dot(buf[i, :])

        self.Hg[off:(self.mgpu + off), :] = buf.to(self.cpu)
        del buf

    def mul(self, x):
        x = x.to(self.gpu)
        res = self.damp * x
        for off in range(0, self.m, self.mgpu):
            Hg = self.Hg[off:(self.mgpu + off), :].to(self.gpu)
            mul = Hg.matmul(x) / self.denom[off:(self.mgpu + off)]
            res -= mul.matmul(Hg)
            del Hg
        torch.cuda.empty_cache()
        return res

    def diag(self):
        res = self.damp * torch.ones(self.d, device=self.gpu, dtype=self.dtype)
        for off1 in range(0, self.m, self.mgpu):
            Hg = self.Hg[off1:(self.mgpu + off1), :].to(self.gpu)
            for i in range(self.mgpu):
                res -= (Hg[i, :] ** 2) / self.denom[i + off1]
            del Hg
        torch.cuda.empty_cache()
        return res
