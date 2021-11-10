## Power allocation models
# ref: Eisen, Mark, et al. "Learning optimal resource allocations in wireless systems." IEEE Transactions on Signal Processing 67.10 (2019): 2775-2790.

import torch
# torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import sysmodels as sysm
from utils import edge_index_batch

#############################################
#####   BASIC MODELS W/O CONSTRAINTS   ######
#############################################

class basic_mlp_nc(nn.Module): # no constraints
    def __init__(self, in_size, out_size, h_sizes, activs=None, dropout=0.5, **extra):
        super(basic_mlp_nc, self).__init__()
        
        # activation functions
        activations = {'elu': nn.ELU(),'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid()}
        activs = ['relu']*len(h_sizes) if activs is None else activs
        
        # modulelist container for hidden layers
        self.hidden = nn.ModuleList()
        self.hidden.append(nn.Linear(in_size, h_sizes[0]))
        self.hidden.append(activations[activs[0]])
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
            self.hidden.append(activations[activs[k+1]])
            self.hidden.append(nn.Dropout(dropout))
        self.hidden.append(nn.Linear(h_sizes[k+1], out_size))
        self.hidden.append(activations[activs[-1]])
        
    def forward(self, x):
        for i, hidden in enumerate(self.hidden):
            x = hidden(x)
        return x

    
class basic_gcn_nc(torch.nn.Module):
    def __init__(self, in_size, out_size, h_sizes, activs=None, dropout=0.5, **extra):
        super(basic_gcn_nc, self).__init__()
        
        # activation functions
        activations = {'elu': nn.ELU(), 'relu': nn.ReLU(), 'selu': nn.SELU(), 
                       'sigmoid': nn.Sigmoid(), 'linear': nn.Linear(out_size, out_size), 'none': nn.Identity()}
        activs = ['relu']*len(h_sizes) if activs is None else activs
        self.activs = nn.ModuleList([activations[a] for a in activs])
        self.dp = dropout
        
        # modulelist container for hidden layers
        hidden_sizes = [in_size, *h_sizes, out_size]
        self.hidden = nn.ModuleList()
        for k in range(len(hidden_sizes)-1):
            self.hidden.append(GCNConv(hidden_sizes[k], hidden_sizes[k+1]))
        
    def forward(self, x, edge_index, edge_weights):
        for i, (hidden, activa) in enumerate(zip(self.hidden, self.activs)):
            x = hidden(x, edge_index, edge_weight=edge_weights)
            x = F.dropout(x, p=self.dp, training=self.training)
            x = activa(x)
        return x
        
################################################
#####   AGGREGATED MODELS (MLP/GCN/USCA)  ######
################################################

class MLP_ChPm_PD(nn.Module): 
    """
        MLP with Channel parameters and Pmax as input (primal dual method)
    """
    def __init__(self, in_size, out_size, h_sizes, activs=None, dropout=0.5, **extra):
        super().__init__()
        self.model = basic_mlp_nc(in_size, out_size, h_sizes, activs, dropout)
        self.size  = out_size
    
    def forward(self, Hx, **extra):
        channels , pmax = Hx[:,self.size:] , Hx[:,[-1]]
        pt = self.model(channels) # already sigmoid-ed
        pt = torch.clamp(pt*pmax, min=1e-12)
        return pt
    
    
class GCN_ChPt_PD(nn.Module):
    """
        GCN with channel parameters as edge weights and pt as node signals
    """
    def __init__(self, in_size, out_size, h_sizes, activs=None, dropout=0.5, **extra):
        super().__init__()
        self.model = basic_gcn_nc(in_size, out_size, h_sizes, activs, dropout)
        self.size  = extra['num_users']
        self.ei    = extra['edge_index']
        self._ei_batch = None
        
    def forward(self, Hx, edge_index=None,**extra): 
        # process the index
        if edge_index is None:
            edge_index = self.ei 
        if 1:#self._ei_batch is None:
            self._ei_batch = edge_index_batch(edge_index, Hx.shape[0], self.size, Hx.device)
        
        p_init = Hx[:,:self.size].reshape(-1,1)
        pmax = Hx[:,[-1]].reshape(-1,1) 
        edge_weights_batch = Hx[:,self.size:-1].reshape(-1) 
        pt = self.model(p_init, self._ei_batch, edge_weights_batch).reshape(-1, self.size)
        pt = torch.clamp(pt*pmax, min=1e-12)
        return pt
    
        
##########################################
#####   PRIMAL-DUAL MODELS (FINAL)  ######
##########################################
    
class WirelessFedL_PrimalDual(nn.Module):
    def __init__(self, model, users, k, constraints, device):
        super(WirelessFedL_PrimalDual,self).__init__()
        self.device = device
        self.model = model
        self.k = torch.ones(users, device=device)/users if k is None else k
        base = torch.zeros((1, users), device=device) # default: requires_grad=False
        
        assert 'q' in constraints
        self.vars, self.lambdas, self.cfunctions = {},{},{}
        for kc in constraints:
            self.vars[kc] = base.clone()
            self.lambdas[kc] = base.clone()
            self.cfunctions[kc] = sysm.get_utility_func(kc)
        self.l_p = 0.
        self.l_d = 0.
        self.Ef = {}
#         self.init()

    def allocate(self, Hx_dir, **extra):
        pt = self.model(**Hx_dir) 
        if self.training:
            return pt
        else:
            return pt*(pt>1e-12)        
        
    def forward(self, Hx_dir, B, m, k=None):
        Hx = Hx_dir['Hx']
        pt = self.model(**Hx_dir) 
               
        if k is None:
            k = self.k
            
        q = sysm.g0(pt, Hx, m)
        self.l_p = sysm.g(q, k).mean(0)

        l_d = 0.
        for kc, f in self.cfunctions.items():
            if kc == 'q':
                Ef = f(pt, Hx, m)
                self.Ef[kc] = Ef.mean(dim=0,keepdim=True)
            else:
                Ef = f(pt, Hx, B)
                mask = Ef>0
                self.Ef[kc] = (Ef*mask).sum(dim=0)/mask.sum(dim=0,keepdim=True)                
            l_d += self.lambdas[kc] @ (self.Ef[kc] - self.vars[kc]).T  
        self.l_d = torch.squeeze(l_d)
        
        self.lagr = self.l_p + self.l_d      
        
        if self.training:
            return pt
        else:
            return pt*(pt>1e-12)  
        
    def update(self, stepsizes, mins, k=None):
        ss_q, ss_c, ss_e, ss_lq, ss_lc, ss_le = stepsizes
        
        if k is None:
            k = self.k 
            
        for kc in self.cfunctions.keys():  # {q, c, e}
            if kc == 'q':
                self.vars[kc] = self.vars[kc] + stepsizes[kc] * ( k - self.lambdas[kc])
            else:
                self.vars[kc] = torch.clip(
                    self.vars[kc] + stepsizes[kc] * ( - self.lambdas[kc]), 
                    min = mins[kc]
                )
            self.lambdas[kc] = torch.relu( 
                self.lambdas[kc] - stepsizes['l'+kc] * (self.Ef[kc] - self.vars[kc]) 
            )
            
        self.detach()
        
    def detach(self):
        for kc in self.cfunctions.keys():
            self.vars[kc].detach_()
            self.lambdas[kc].detach_()
        
    def init(self):
        for kc in self.cfunctions.keys():
            nn.init.kaiming_uniform_(self.vars[kc])
            nn.init.kaiming_uniform_(self.lambdas[kc])
        