import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import fsolve

from utils import decouple_input, scale_to_range


## Wireless system model

class WirelessModel(object):
    def __init__(self, bandwidth=1):
        self.hc = 10**(-27) * 40 * 10**18  # hardware constants
        self.bw = bandwidth                # uplink bandwidth
        
    @staticmethod
    def packet_error_rate(h, I, P, noise=10**(-14), m=0.023):
        """
        Packet error rate of each user over each RB
        inputs: 
            h - channel gains, 
            I - interference, 
            P - power
        return:
            q - packet_error_rate
        """
        q = 1 - np.exp( -m * (I + noise) / (P*h) )
        if q.ndim==3:
            q = q.mean(0)
        return q

    @staticmethod
    def signal_interf_noise_ratio(h, I, P, noise=10**(-14)):
        """
        SINR (Signal-to-interference-plus-noise ratio) of each user over each RB 
        inputs: 
            h - channel gains, 
            I - interference, 
            P - power
        return:
            SINR - packet_error_rate
        """ 
        sinr = (P*1*h) / (I+noise);  # SINR of each user over each RB
        return sinr.astype(np.complex) 
        
    def data_rate(self, sinr, bandwidth=None):
        if bandwidth is None:
            bandwidth = self.bw
        rateu = np.log2(1+sinr)
        if rateu.ndim == 3:
            rateu = rateu.mean(0) # take expected value
        rateu *= bandwidth;      # Uplink data rate of each user over each RB
        return rateu

    def consumed_energy(self, P, data_bits, delay_up):
        train_e = self.hc * data_bits   # training energy
        trans_e = np.mean(P) * delay_up  # transmission energy
        return train_e + trans_e
    
    @staticmethod
    def power_control_i(pmax_i, e_trans_i, data_bits, h_i, I_n, noise=10**(-14)):
        def efunc(p):
            return p * data_bits/ np.log2(1+p*1*h_i / (I_n+noise)).mean() - e_trans_i
        result = fsolve(efunc, x0=pmax_i)
        pi = min(pmax_i, max(result.item(),1e-8))
        return pi
    
    @staticmethod
    def power_interference(beta, P):
        I = np.zeros_like(P)
        u, r = P.shape
        for rb in range(r):
            for i in range(u):
                for j in range(u):
                    if j!=i:
                        I[i,rb] += beta[i,j]*P[j,rb]
        return I
        
    
## Federated learning system model

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size) 
        )
    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = self.model(x)
        return x
    

class patternnet(object):
    def __init__(self, input_size, hidden_size, output_size, 
                 criterion=nn.BCEWithLogitsLoss, learning_rate=1e-3, initFunc=None, name_string=''):
        super().__init__()
        self.model = FNN(input_size, hidden_size, output_size)
        self.init = initFunc
        self.reset_parameters()
        
        self.name = name_string
        self.criterion = criterion()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)#, weight_decay=1e-6)
        
    def custom_initialize(self):
        for layers in self.model.children():
            for li, layer in enumerate(layers):
                if hasattr(layer, 'reset_parameters'):
                    if li+1 < len(layers):
                        inp_active = [-2,2] # tansig activation for hidden layers
                    else:
                        inp_active = [-np.Inf, np.Inf] # softmax for output layer
                    ci, cn = layer.in_features, layer.out_features
                    w,b = self.init(ci, cn , inp_active, np.asarray([-1,1]*ci).reshape(-1,2))
                    layer.weight.data = torch.from_numpy(w).float()
                    layer.bias.data = torch.from_numpy(b).float()
             
    def default_initialize(self):
        for layers in self.model.children():
            for layer in layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def reset_parameters(self):
        if self.init is None:
            self.default_initialize()
        else:
            self.custom_initialize()

                    
## Performance and utility functions

def sinr_i(pt, Hx, *args):
    """ pointwise data rate """
    nue = pt.shape[-1]
    _, de_h, de_pmax = decouple_input(Hx, nue)
    y_pred_s = scale_to_range(pt, [0, de_pmax.view(-1)])
    if not torch.all(y_pred_s==pt):
        warnings.warn('prediction out of range! clipping...')
    pt = y_pred_s
   
    s = de_h*pt.view((-1, 1, nue)) # shape: (4,4) * (4,) --> (4,4)
    direct = s.diagonal(dim1=-2, dim2=-1)
    ifn = torch.sum(s, axis=-1) - direct + 1
    sinr = direct/ifn
    return sinr

def f_trans_success_rate(pt, Hx, m):
    sinr = sinr_i(pt, Hx)
    q = 1-torch.exp(-m/sinr) # error
    return 1-q # correctness

def f_data_rate(pt, Hx, B):
    sinr = sinr_i(pt, Hx)
    c = B*torch.log(1+sinr)
    return c

def f_energy(pt, Hx, B):
    sinr = sinr_i(pt, Hx)
    e = B*torch.log(1+sinr)/pt
    return e

# aliases
f1 = f_trans_success_rate
f2 = f_data_rate
f3 = f_energy


def get_utility_func(m):
    func_dict = {
        'q': f_trans_success_rate,
        'c': f_data_rate,
        'e': f_energy
    }
    return func_dict[m]


def g0(pt, Hx, m):
    return f_trans_success_rate(pt, Hx, m)

def g(q, k=None):
    if k is None:
        k=torch.ones(q.shape[1]).to(q.device)
    obj = k * q / k.sum()
    return obj.sum(1)   
                
                    