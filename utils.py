import os
import h5py
import warnings
import numpy as np
from datetime import datetime
from prettytable import PrettyTable

import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
    
import builtins as __builtin__

# np.random.seed(0)
# torch.manual_seed(0)

    
def print(*args, **kwargs):
    # My custom print() function: Overload print function to get time logged
    __builtin__.print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), end = ' | ')
    return __builtin__.print(*args, **kwargs)

def print_update(msg, pbar=None):
    if pbar is not None:
        pbar.write(msg)
    else:
        print(msg)        

def reset_model_parameters(m):
    # check if m is iterable or if m has children
    # base case: m is not iterable, then check if it can be reset; if yes --> reset; no --> do nothing

    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()
    else:
        for l in m.children():
            reset_model_parameters(l)

def count_parameters(model, verbose=0):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad: 
            param = parameter.numel()
            table.add_row([name, param])
            total_params+=param
    if verbose >= 2:
        print(table)
    if verbose >= 1:
        print(f"Total Trainable Params: {total_params}")
    return total_params


def scale_to_range(x, constraints):
    if constraints is None:
        return x

    device = x.device
    
    lo, hi = constraints
    n,d = x.shape
    
    if not hasattr(lo, "__len__"): # hi is a scalar
        lo *= torch.ones(n).view(-1).to(device)
        
    if not hasattr(hi, "__len__"): # hi is a scalar
        hi *= torch.ones(n).view(-1).to(device)
        
    lo = lo.repeat(d).view((d,-1)).T
    hi = hi.repeat(d).view((d,-1)).T
    x_clip = torch.max(torch.min(x, hi), lo)
    
    return x_clip


def init_p(pmax, nue, method='full'):
    full_init = np.tile(pmax,(nue,1)).T
    if method=='full':
        return full_init
    rand_init = np.random.uniform(low=0.0, high=full_init)
    return rand_init


def decouple_input(x, n):
    cpt1, cpt2, cpt3 = -(n+n**2+1), -(n**2+1), -1  # with/without y_pred as start
    de_pmax = x[:,cpt3]
    de_h = x[:,cpt2:cpt3].view(-1, n , n )
    de_p = x[:,:cpt2]
        
    if x.shape[1]!=-cpt1 and x.shape[1]!=-cpt2 :
        raise ValueError('check size of input!')
    return de_p, de_h, de_pmax


def edge_index_batch(ei, bs, nu, dev):
    shift = torch.vstack([torch.arange(bs)*nu,]*nu**2).T.reshape(-1).repeat(1, 2).view(2,-1).long().to(dev) 
    ei_batch = ei.repeat(1, bs)+shift
    return ei_batch
        

## util functions for training and evaluation

def train(net, dataloader, epochs):
#     print(f'Training {net.name} ...')
    for ep in range(epochs):
        for data, target in dataloader:
            #data, target = data.to(device), target.to(device)
            net.optimizer.zero_grad()
            logits = net.model(data)
            loss = net.criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.model.parameters(), 1.)
            net.optimizer.step()
    return net
            
            
def predict(net, dataloader):
    logits = []
    for data, target in dataloader:
        logits.append(net.model(data))
    logits = torch.cat(logits)
    prob = F.softmax(logits, dim=-1)
    pred = prob.argmax(dim=-1)
    return prob, pred


def weighted_selected_error_eval(w, q, reduce='mean'):
    weighted_e = w*q
    selected_i = q<1
    vals = (w*q)[q<1].sum(-1) / np.tile(w,(q.shape[0], 1))[q<1].sum(-1)
    if reduce.lower()=='none':
        val = vals
    elif reduce.lower() == 'mean':
        val = vals.mean()    
    return val


def logs(logdir, model, inputdir, lkey, save=None, vio=None):
    model.eval()
    pt = model(**inputdir)
    
    def append_as_dict_vals(d, k, v):
        if k in d:
            d[k].append(v)
        else:
            d[k] = [v]
        return d
    
    lp,ld =  model.l_p.item(), model.l_d.item()
    append_as_dict_vals(logdir[lkey], 'l_p', lp)
    append_as_dict_vals(logdir[lkey], 'l_d', ld)
    vflag = False
    for kc, ef in model.Ef.items():
        ev = ef.cpu().detach().numpy()
        append_as_dict_vals(logdir[lkey], 'Ef_'+kc, ev)
        if (vio is not None) and (kc in vio) and (vio[kc] is not None) and (ev.mean()<vio[kc]):
            vflag = True
            
    if not vflag: 
        # if best performance, and satisfy constraints
        append_as_dict_vals(logdir[lkey], '_l_p_sat', lp)
        if np.all(logdir[lkey]['_l_p_sat'][-1] >= np.array(logdir[lkey]['_l_p_sat'])) and save:
            torch.save(model, save + 'model_pd.pt')
    else:
        if np.all(logdir[lkey]['l_p'][-1] >= np.array(logdir[lkey]['l_p'])) and save:
            torch.save(model, save + 'model_pd-vio.pt')
        
    return logdir


def initnw(ci, cn, inp_active, inp_minmax):
    """
    Nguyen-Widrow initialization function 
    (adapted from: https://pythonhosted.org/neurolab/_modules/neurolab/init.html)
    
    :Parameters:
        ci: int
            Number of inputs (feature dimension)
        cn: int
            Number of neurons (hidden dimension)
        inp_active: list input active range [min, max]
            Input active range
            tansig  --> inp_active = [-2, 2]
            softmax --> inp_active = [-np.Inf, np.Inf] = [-1, 1]
        inp_minmax: minmax: list ci x 2
            Range of input value
    """
    w_fix = 0.7 * cn ** (1. / ci)
    w_rand = np.random.rand(cn, ci) * 2 - 1
    # Normalize
    if ci == 1:
        w_rand = w_rand / np.abs(w_rand)
    else:
        w_rand = np.sqrt(1. / np.square(w_rand).sum(axis=1).reshape(cn, 1)) * w_rand

    w = w_fix * w_rand
    b = np.array([0]) if cn == 1 else w_fix * np.linspace(-1, 1, cn) * np.sign(w[:, 0])

    # Scaleble to inp_active
    amin, amax  = inp_active
    amin = -1 if amin == -np.Inf else amin
    amax = 1 if amax == np.Inf else amax

    x = 0.5 * (amax - amin)
    y = 0.5 * (amax + amin)
    w = x * w
    b = x * b + y

    # Scaleble to inp_minmax
    minmax = inp_minmax
    minmax[np.isneginf(minmax)] = -1
    minmax[np.isinf(minmax)] = 1

    x = 2. / (minmax[:, 1] - minmax[:, 0])
    y = 1. - minmax[:, 1] * x
    b = np.dot(w, y) + b
    w = w * x
    
    return w,b

     
## Load data

def load_data_unsup(dpath, **kwargs):
    # Choice of parameters is with reference to https://github.com/bmatthiesen/deep-EE-opt/blob/062093fde6b3c6edbb8aa83462165265deefce1a/src/globalOpt/run_wsee.py#L30
    extract_args = lambda a, k: a if not k in kwargs else kwargs[k]
    PdB = extract_args(np.array(range(-40,10+1,1)), 'PdB')   
    
    mu = extract_args(4.0, 'mu')
    Pc = extract_args(1.0, 'Pc')
    hxp = extract_args(True, 'hxp')
    num_stab = extract_args(0., 'num_stab')    
    
    Plin = 10**(PdB/10)
    if hxp:
        Ph = Plin
    else:
        Ph = torch.empty(Plin.shape).fill_(1)
    
    X = []
    with h5py.File(dpath, "r") as handle:
        Hs = handle['input']["channel_to_noise_matched"]
        
        ns, nu, _ = Hs.shape # eg:(1000,4,4)

        for hidx in range(ns):
            edge_index, h = dense_to_sparse(torch.from_numpy(Hs[hidx].astype(float)))
            h += num_stab
            
            x1 = np.hstack([(h.reshape((-1,1))*Ph).T, # -->(h1p1, h2p1, h3p1, ...)
                             Plin.reshape(-1,1)])
            X.append( x1 )
            
        cinfo = {'mu': mu,
                 'Pc': Pc,
                 'edge_index': edge_index} # or wsee 

    X = np.concatenate((X))
    X = X[~np.any(np.isnan(X),-1)]
    y = np.full([X.shape[0], nu], np.nan)
    
    return X, y, cinfo


