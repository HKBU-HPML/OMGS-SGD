# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import time
from compression import compressors

compstr = 'topk'
compressor = compressors[compstr]

def bench_single_topk(n, ratio, num_iters):
    tensor = torch.rand(n).float().cuda()
    k = int(ratio * n)
    values, indexes = torch.topk(torch.abs(tensor.data), k=k, sorted=False)
    torch.cuda.synchronize()
    stime = time.time()
    for i in range(num_iters):
        values, indexes = torch.topk(torch.abs(tensor.data), k=k, sorted=False)
    torch.cuda.synchronize()
    etime = time.time()
    time_used = (etime-stime)/num_iters
    return time_used

def bench():
    ns = range(2**10, 2**20, 1024) 
    ns = ns+range(2**20, 2**29, 2**20) 
    ratio = 0.001
    for n in ns:
        num_iters = 50
        if n > 2**19:
            num_iters = 10
        t = bench_single_topk(n, ratio, num_iters)
        print('%d,%f'%(n,t))



if __name__ == '__main__':
    bench()
