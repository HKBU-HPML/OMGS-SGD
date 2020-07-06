# -*- coding: utf-8 -*-
from __future__ import print_function
from settings import logger
import torch
import numpy as np
import time


class NoneCompressor():
    @staticmethod
    def compress(tensor, name=None):
        return tensor, tensor.dtype

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor 
        return z 


class TopKCompressor():
    """
    Sparse Communication for Distributed Gradient Descent, Alham Fikri Aji et al., 2017
    """
    residuals = {}
    sparsities = []
    zero_conditions = {}
    values = {} 
    indexes = {} 
    c = 0
    t = 0.
    name = 'topk'

    @staticmethod
    def clear():
        TopKCompressor.residuals = {}
        TopKCompressor.sparsities = []
        TopKCompressor.zero_conditions = {}
        TopKCompressor.values = {} 
        TopKCompressor.indexes = {} 

    @staticmethod
    def compress(tensor, name=None, sigma_scale=2.5, ratio=0.05):
        with torch.no_grad():
            if name not in TopKCompressor.residuals:
                TopKCompressor.residuals[name] = torch.zeros_like(tensor.data)
            # top-k solution
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.data.add_(TopKCompressor.residuals[name].data)

            values, indexes = torch.topk(torch.abs(tensor.data), k=k)
            values = tensor.data[indexes]
            TopKCompressor.residuals[name].data = tensor.data + 0.0
            TopKCompressor.residuals[name].data[indexes] = 0. 

            TopKCompressor.values[name] = values
            TopKCompressor.indexes[name] = indexes
            return tensor, indexes, values

    @staticmethod
    def get_residuals(name, like_tensor):
        if name not in TopKCompressor.residuals:
            TopKCompressor.residuals[name] = torch.zeros_like(like_tensor.data)
        return TopKCompressor.residuals[name]

    @staticmethod
    def add_residuals(included_indexes, name):
        with torch.no_grad():
            residuals = TopKCompressor.residuals[name]
            if type(included_indexes) is np.ndarray:
                indexes_t = torch.from_numpy(included_indexes).to(device=residuals.device).long()
            else:
                indexes_t = included_indexes
            values = TopKCompressor.values[name]
            values.data[indexes_t] = 0.0
            residuals.data[TopKCompressor.indexes[name]] += values.data

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor 
        return z 


compressors = {
        'topk': TopKCompressor,
        'none': NoneCompressor,
        None: NoneCompressor
        }
