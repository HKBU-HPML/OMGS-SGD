import hashlib
import time
import os
import numpy as np


def gen_random_id():
    id_ = hashlib.sha256()
    id_.update(str(time.time()))
    return id_.hexdigest()

def create_path(relative_path):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, relative_path)
    if not os.path.isdir(filename):
        try:
            #os.mkdir(filename)
            os.makedirs(filename)
        except:
            pass

def update_fontsize(ax, fontsize=12.):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)

def autolabel(rects, ax, label, rotation=90):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_y() + rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            label,
            ha='center', va='bottom', rotation=rotation)

def topk(tensor, k):
    indexes = np.abs(tensor).argsort()[-k:]
    return indexes, tensor[indexes]

def predict_density_with_size_and_computation(m, comp_time, P):
    alpha = 4*0.436e-3
    beta =  4*9e-6*1e-3
    def _denseallreduce_model(P, m):
        return 2*(P-1)*alpha + 2* (P-1)/P * m * beta

    def _sparseallreduce_model(P, m, rho=0.001):
        return np.log2(P) + 2 * (P - 1) * rho * m * beta

    def _proper_rho_with_sparse_allreduce(P, m, comp_time):
        rho = 0.001
        t = comp_time - np.log2(P) * alpha 
        if t <= 0:
            return rho 
        rho = t/ (2*(P-1)*beta*m)
        if rho > 1.0:
            rho = 0.05
        rho = max(rho, 0.001)
        return rho
    return 0.001

def predict_allreduce_time_with_size(alpha, beta, size, P):
    return alpha + beta * size 

def force_insert_item(d, key, val):
    if key not in d:
        d[key] = []
    d[key].append(val)

s=2.18896957e-10 #P102-100
GbE_multi_p_ab = {
        2: (1.6e-3, 1.0e-8),
        4: (2.7e-3, 1.3e-8),
        8: (4.0e-3, 1.5e-8),
        16: (1.1e-2, 1.7e-8)
        }

def topk_perf_model(x, s=s):
    """
    x is the number of parameters
    Return: s * x * log2(x)
    """
    if x == 0.0:
        return 0.0
    return s * x * np.log2(x)

def allgather_perf_model(x, P, eth='GbE'):
    """
    x is the number of parameters
    Return: t = a + b * x
    """
    multi_p_ab = GbE_multi_p_ab
    a, b = multi_p_ab[P]
    if x == 0.0:
        return 0.0
    return (a + b * x * P * 4) * 2

