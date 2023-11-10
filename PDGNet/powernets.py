## Power allocation models
# ref: Eisen, Mark, et al. "Learning optimal resource allocations in wireless systems." IEEE Transactions on Signal Processing 67.10 (2019): 2775-2790.

import torch
# torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import sysmodels as sysm
from utils import edge_index_batch, torch_clip_min_tensor, get_local_didx_list

PMIN_STAB = 1e-20
PMIN_COND = 1e-10


def norm(x):
    # normalise x to range [0,1]
    nom = (x - x.min()) #* 2.0
    denom = x.max() - x.min()
    return  nom/denom #- 1.0

def sigmoid(x, k=0.1):
    # sigmoid function
    # use k to adjust the slope
    s = 1 / (1 + torch.exp(-(x*2-1) / k)) 
    return s

class zInitMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(1),
        )
#         self.model.apply(self.init_zero)
        
    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = self.model(x)
        return x
    
    @staticmethod
    def init_zero(m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.)
            m.bias.data.fill_(0.)
            

class DataEvalSupCls(nn.Module):
    def __init__(self, model, datanums):
        super().__init__()
        self.model = model
        self.device = next(self.model.parameters()).device
        self.data_indices = get_local_didx_list(datanums)
        
    def forward(self, data_loader):
        qidx = []
        for x, y in data_loader:
            yp = self.model(x.to(self.device))
            loss = 1 / nn.CrossEntropyLoss(reduction='none')(yp, y.to(self.device))
            qidx.append(loss)
        q_indicators = torch.cat(qidx)    
        weights = torch.zeros(len(self.data_indices)).to(self.device)
        for i, di in enumerate(self.data_indices):
            weights[i] = q_indicators[di].sum()/q_indicators.sum()
        return weights
    
#############################################
#####   BASIC MODELS W/O CONSTRAINTS   ######
#############################################

class basic_mlp_nc(nn.Module): # no constraints
    def __init__(self, in_size, out_size, h_sizes, activs=None, dropout=0.5, **extra):
        super(basic_mlp_nc, self).__init__()
        
        # activation functions
        activations = {'elu': nn.ELU(),'relu': nn.ReLU(), 'leakyrelu': nn.LeakyReLU(), 'sigmoid': nn.Sigmoid()}
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
        x = torch.log10(x) # for mlp: convetr input to dbw
        for i, hidden in enumerate(self.hidden):
            x = hidden(x)
        return x

    
class basic_gcn_nc(torch.nn.Module):
    def __init__(self, in_size, out_size, h_sizes, activs=None, dropout=0.5, **extra):
        super(basic_gcn_nc, self).__init__()
        
        # activation functions
        activations = {'elu': nn.ELU(), 'relu': nn.ReLU(), 'selu': nn.SELU(), 'leakyrelu': nn.LeakyReLU(),
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
        pt_scaled = self.model(channels) # already sigmoid-ed
        pt = torch.clamp(pmax * pt_scaled, min=PMIN_STAB)
        return pt
    
    
class GCN_ChPt_PD(nn.Module):
    """
        GCN with channel parameters as edge weights and pt as node signals
    """
    def __init__(self, in_size, out_size, h_sizes, activs=None, dropout=0.5, **extra):
        super().__init__()
        self.model = basic_gcn_nc(in_size, out_size, h_sizes, activs, dropout)
        
        self.net_size  = extra['num_users']
        self.ei        = extra['edge_index']
        self._ei_batch = None
        
        self.sig_size = in_size
        
    def forward(self, Hx, edge_index=None,**extra): 
        """ 
            Hx: [(w_1, w_2, ..., w_8), p_in_1, p_in_2, ..., p_in_8, h_1, h_2, ..., h_64, pmax] 
        """
        # process the index
        if edge_index is None:
            edge_index = self.ei 
        if 1:#self._ei_batch is None:
            self._ei_batch = edge_index_batch(edge_index, Hx.shape[0], self.net_size, Hx.device)
                
        in_size = int(self.net_size * self.sig_size)
        p_in = Hx[:,:in_size].reshape(self.sig_size,-1).T
        pmax = Hx[:,[-1]].reshape(-1,1) 
        edge_weights_batch = Hx[:,in_size:-1].reshape(-1) 
        pt_scaled = self.model(p_in, self._ei_batch, edge_weights_batch).reshape(-1, self.net_size)
        pt = torch.clamp(pmax * pt_scaled, min=PMIN_STAB)
        return pt
    
    
##########################################
#####   PRIMAL-DUAL MODELS (FINAL)  ######
##########################################
    
class WirelessFedL_PrimalDual(nn.Module):
    def __init__(self, model, kw, users, constraints, device):
        super(WirelessFedL_PrimalDual,self).__init__()
        self.device = device
        self.model = model
        self.size = users
        self.k = kw#torch.ones(users, device=device)/users if kw is None else kw
        base = torch.zeros((1, users), device=device) # default: requires_grad=False
        
        constraints = [c for c in constraints if c in ['q','c','e']]
        assert 'q' in constraints
        self.vars, self.lambdas, self.cfunctions = {},{},{}
        for kc in constraints:
            self.vars[kc] = base.clone()
            self.lambdas[kc] = base.clone()
            self.cfunctions[kc] = sysm.get_utility_func(kc)
        self.l_p = 0.
        self.l_d = 0.
        self.loss_bp = None
        self.Ef = {}
        
#         self.init_prime()
#         self.init_dual()


    def allocate(self, Hx_dir, **extra):
        pt = self.model(**Hx_dir) 
        return pt*(pt>PMIN_COND)
    
    def get_user_weights(self, kw):
        if self.k is None and kw is None:
            k = None 
        elif self.k is None:
            k = kw
        else:
            k = self.k 
        return k
        
    def forward(self, Hx_dir, B, m, kw=None):
        Hx = Hx_dir['Hx']
        pt = self.model(**Hx_dir) 
#         assert not torch.all(torch.isfinite(pt))
        # Get the k_weights
        if (k := self.get_user_weights(kw)) is None:
            k = Hx[0,:self.size]
        
        self.l_p = sysm.g(self.vars['q'], k).mean(0)

        l_d = 0.
        for kc, f in self.cfunctions.items():
            if kc == 'q':
                Ef = f(pt, Hx, m)
                self.Ef[kc] = Ef.mean(dim=0,keepdim=True)             
            else:
                Ef = f(pt, Hx, B)
                mask = pt > PMIN_COND #Ef > 0
                self.Ef[kc] = (Ef * mask).sum(dim=0)/mask.sum(dim=0,keepdim=True)     
#             print(kc, self.lambdas[kc].shape)
            l_d += self.lambdas[kc] @ (self.Ef[kc] - self.vars[kc]).T  
            
        self.l_d = torch.squeeze(l_d)
        self.lagr = self.l_p + self.l_d  
        
        # loss to back prop
        self.loss_bp = self.l_d #self.lagr | self.l_d
        
        return pt*(pt>PMIN_COND)
    
        
    def update(self, stepsizes, mins, kw=None):
        ss_q, ss_c, ss_e, ss_lq, ss_lc, ss_le = stepsizes
        
        # Get the k_weights
        if (k := self.get_user_weights(kw)) is None:
            raise
            
        for kc in self.cfunctions.keys():  # {q, c, e}
            if kc == 'q':
                self.vars[kc] = self.vars[kc] + stepsizes[kc] * ( k - self.lambdas[kc])
            else: 
                self.vars[kc] = torch.clip(
                    self.vars[kc] + stepsizes[kc] * ( - self.lambdas[kc]), 
                    min = mins[kc]
                )
#                 # Non-uniform constraints handling
#                 self.vars[kc] = torch_clip_min_tensor(
#                     self.vars[kc] + stepsizes[kc] * ( - self.lambdas[kc]), 
#                     min = mins[kc].view(self.vars[kc].shape)
#                 ) 
            self.lambdas[kc] = torch.relu( 
                self.lambdas[kc] - stepsizes['l'+kc] * (self.Ef[kc] - self.vars[kc]) 
            )
        self.detach()
        
    def detach(self):
        for kc in self.cfunctions.keys():
            self.vars[kc].detach_()
            self.lambdas[kc].detach_()
        
    def init_dual(self):
        for kc in self.cfunctions.keys():
#             self.lambdas[kc].data.uniform_(0, .2)
            self.lambdas[kc].data.fill_(0.1)

    def init_prime(self, Hx_dir, B, m):
        Hx = Hx_dir['Hx']
        pt = self.model(**Hx_dir) 
        for kc, cfunc in self.cfunctions.items():
            if kc == 'q':
                self.vars[kc] = cfunc(pt, Hx, m).mean(dim=0, keepdim=True)
            else:
                self.vars[kc] = cfunc(pt, Hx, B).mean(dim=0)

                
                
class WirelessFedL_PrimalDual_Smpl(nn.Module):
    def __init__(self, model, kw, users, constraints, device):
        super(WirelessFedL_PrimalDual_Smpl,self).__init__()
        self.device = device
        self.model = model
        self.size = users
        self.k = kw#torch.ones(users, device=device)/users if kw is None else kw
        base = torch.zeros((1, users), device=device) # default: requires_grad=False
        
        constraints = {c:v for c,v in constraints.items() if c in ['q','c','e']}
        assert 'q' in constraints
        self.vars, self.lambdas, self.cfunctions = {},{},{}
        for kc in constraints:
            if kc == 'q':
                self.vars[kc] = base.clone()
            else:
                self.vars[kc] = constraints[kc]
                
            self.lambdas[kc] = base.clone()
            self.cfunctions[kc] = sysm.get_utility_func(kc)
        self.l_p = 0.
        self.l_d = 0.
        self.loss_bp = None
        self.Ef = {}
        
        self.init_prime()
        self.init_dual()


    def allocate(self, Hx_dir, **extra):
        pt = self.model(**Hx_dir) 
        return pt*(pt>PMIN_COND)
    
    def get_user_weights(self, kw):
        if self.k is None and kw is None:
            k = None 
        elif self.k is None:
            k = kw
        else:
            k = self.k 
        return k
        
    def forward(self, Hx_dir, B, m, kw=None):
        Hx = Hx_dir['Hx']
        pt = self.model(**Hx_dir) 
#         assert not torch.all(torch.isfinite(pt))
        # Get the k_weights
        if (k := self.get_user_weights(kw)) is None:
            k = Hx[0,:self.size]
        
        self.l_p = sysm.g(self.vars['q'], k).mean(0)

        l_d = 0.
        for kc, f in self.cfunctions.items():
            if kc == 'q':
                Ef = f(pt, Hx, m)
                self.Ef[kc] = Ef.mean(dim=0,keepdim=True)             
            else:
                Ef = f(pt, Hx, B)
                mask = pt > PMIN_COND #Ef > 0
                self.Ef[kc] = (Ef * mask).sum(dim=0)/mask.sum(dim=0,keepdim=True)     
#             print(kc, self.lambdas[kc].shape)
            l_d += self.lambdas[kc] @ (self.Ef[kc] - self.vars[kc]).T  
            
        self.l_d = torch.squeeze(l_d)
        self.lagr = self.l_p + self.l_d  
        
        # loss to back prop
        self.loss_bp = self.l_d 
        
        return pt*(pt>PMIN_COND)
    
        
    def update(self, stepsizes, mins, kw=None):
        ss_q, ss_c, ss_e, ss_lq, ss_lc, ss_le = stepsizes
        
        # Get the k_weights
        if (k := self.get_user_weights(kw)) is None:
            raise
            
        for kc in self.cfunctions.keys():  # {q, c, e}
            if kc == 'q':
                self.vars[kc] = self.vars[kc] + stepsizes[kc] * ( k - self.lambdas[kc])
#             else: 
#                 self.vars[kc] = torch.clip(
#                     self.vars[kc] + stepsizes[kc] * ( - self.lambdas[kc]), 
#                     min = mins[kc]
#                 )
# #                 # Non-uniform constraints handling
# #                 self.vars[kc] = torch_clip_min_tensor(
# #                     self.vars[kc] + stepsizes[kc] * ( - self.lambdas[kc]), 
# #                     min = mins[kc].view(self.vars[kc].shape)
# #                 ) 
            self.lambdas[kc] = torch.relu( 
                self.lambdas[kc] - stepsizes['l'+kc] * (self.Ef[kc] - self.vars[kc]) 
            )
        self.detach()
        
    def detach(self):
        for kc in self.cfunctions.keys():
            self.vars['q'].detach_()
            self.lambdas[kc].detach_()
        
    def init_dual(self):
        for kc in self.cfunctions.keys():
#             self.lambdas[kc].data.uniform_(0, .2)
            self.lambdas[kc].data.fill_(0.1)

    def init_prime(self, Hx_dir, B, m):
        Hx = Hx_dir['Hx']
        pt = self.model(**Hx_dir) 
        for kc, cfunc in self.cfunctions.items():
            if kc == 'q':
                self.vars[kc] = cfunc(pt, Hx, m).mean(dim=0, keepdim=True)
#             else:
#                 self.vars[kc] = cfunc(pt, Hx, B).mean(dim=0)        
        
  