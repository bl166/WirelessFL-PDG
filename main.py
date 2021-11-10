import os
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# SET YOUR DATA & PROJECT ROOT DIR HERE
PROJ_RT = os.getcwd()
DATA_RT = os.getcwd()+'/data/' 

from utils import *
from powernets import MLP_ChPm_PD, GCN_ChPt_PD, WirelessFedL_PrimalDual


device = torch.device('cuda') # cuda | cpu

L = 8  # 6, 16, 24, 32
M_BS = 1
NR = 10
DIST = {'tr'  : 'Alessio',  # Alessio | HataSuburban | HataSuburban-noSF | HataUrban | HataUrban-noSF
        'val' : 'Alessio',
        'test': 'Alessio'}
PDB = np.array(range(-40,10+1,1))
B = 1
M = 0.023

MODEL = GCN_ChPt_PD # MLP_ChPm_PD | GCN_ChPt_PD 

if 'MLP' in str(MODEL):
    AFLAG = 'MLP'
    in_size , out_size = L**2+1 , L
    inner_architect = [
            {'h_sizes': [128, 256, 64, 16, 8],
             'activs': ['elu', 'elu', 'elu', 'elu', 'elu', 'sigmoid']}
    ]
elif 'GCN' in str(MODEL):
    AFLAG = 'GCN'
    in_size , out_size = 1 , 1
    inner_architect = [
            {'h_sizes': [16, 32, 64, 16, 2], 
             'activs': ['elu', 'elu', 'elu', 'elu',  'elu', 'sigmoid']}
    ]
else:
    raise

dropout = 0
l2 = 1e-6
epochs = 1000
save_freq = 10
RSEED= 42

# save results at
RSLT_PATH = lambda intf,pdbm: \
    PROJ_RT+f"/models/{DIST['tr']}_Ant+{NR}_User+{L}_{AFLAG}_Intfx{intf}_Pmax{pdbm:+}_seed+{RSEED}/"


## Data

def get_data(iscale, pmaxi):
    
    indexing = np.delete(np.arange(L**2), np.arange(L**2)[::L+1])

    # add initial pt (max)
    attach_pt = lambda x: torch.from_numpy(np.hstack((init_p(x[:,-1], L, method="full"), x))).float().to(device)

    # move channel info to device
    dict_to_device = lambda x,dev: {k:v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in x.items()}

    X, y, cinfo = {},{},{}
    for phase in ['tr','val','test']:        
        sufix = '' if phase=='tr' else phase+'-'
        dfn = DATA_RT + f"bs+{M_BS}_ant+{NR}/{sufix}channels-{DIST[phase]}-{L}-1000.h5" # train / validation data
        X[phase],y[phase],cinfo[phase] = load_data_unsup(dfn, hxp=False, num_stab=1e-12, PDB=PDB)

        X[phase] = attach_pt(X[phase])
        X[phase][:,L:-1][:,indexing]*=iscale
        
        # with intended pmax
        X[phase] = X[phase][p_shift::len(PDB),:]
        y[phase] = X[phase][:,:L]

        cinfo[phase] = dict_to_device(cinfo[phase], device)
        print(phase, X[phase].shape, y[phase].shape, X[phase][0,L])

    return X,y,cinfo

    
I2C = {1:.6, 2:.4, 4:.2, 8:.2}  # when pmax = -10 dbw
P2C = {0:.2, 10:.4, 20:.5, 30:.6, 40:.6, 50:.6}

# H ( r | r > 0 ) : 
I2C = {1:.8, 2:.6, 4:.3, 8:.2}  # when pmax = -10 dbw
I2C = {1:.7, 2:.5, 4:.3, 8:.1}  # when pmax = -20 dbw
P2C = {0:.25, 10:.55, 20:.7, 30:.8, 40:.8, 50:.8}

    
for i_scale in [1]: # interference scaling: 1 2 4 8
    
#     Rc = I2C[i_scale]
    
    for p_shift in [40]: # from 0, 10, 20, [30], 40, 50
        
        Rc = P2C[p_shift]
    
        save_path = RSLT_PATH(i_scale, PDB[p_shift])
        if not os.path.exists(save_path): os.makedirs(save_path)

        X, y, cinfo = get_data(i_scale, p_shift)

        datanumbers = np.random.RandomState(RSEED).randint(100,1000,L).astype('float')
        k_weights = torch.from_numpy(datanumbers/datanumbers.sum()).float().to(device)

        #
        learning_rate = 1e-3
        update_dict   = lambda ii, cc: {'stepsizes': dict(zip(['q','c','e','lq','lc','le'],
                                                              [learning_rate*.1]*6)),
                                        'mins': {'c':cc*B, 'e':None}, 'k': k_weights}

        for const_c in [Rc]:

            # instantiate model   
            model_alloc = MODEL(num_blocks = 1,  # this is for unfolding
                                num_users  = L,
                                in_size    = in_size, 
                                out_size   = out_size, 
                                **inner_architect[0], 
                                edge_index = None, #cinfo['tr']['edge_index'], 
                                dropout    = dropout).to(device)
            model_pd = WirelessFedL_PrimalDual(model = model_alloc, 
                                               users = L, 
                                               k     = k_weights, 
                                               constraints=['q','c'],
                                               device=device)
            num_params = count_parameters(model_alloc, 1)    
            optimizer = torch.optim.Adam(model_alloc.parameters(), lr=learning_rate, weight_decay=l2)

            # 
            dataset = TensorDataset(X['tr'], y['tr'])
            loader = DataLoader(dataset, batch_size=100, shuffle=True)

            log = {'val':{}, 'test':{}}
            log['val'][const_c], log['test'][const_c] = {},{}

            inputdir_val = {'Hx_dir':{'Hx':X['val'], 'edge_index':cinfo['val']['edge_index']}, 
                           'B':B, 'm':M, 'k':k_weights}
            inputdir_test = {'Hx_dir':{'Hx':X['test'], 'edge_index':cinfo['test']['edge_index']}, 
                           'B':B, 'm':M, 'k':k_weights}
            log['val'] = logs(log['val'], model_pd, inputdir_val, lkey=const_c)
            log['test'] = logs(log['test'], model_pd, inputdir_test, lkey=const_c)

            for ep in range(1000):

                for i, (hx,_) in enumerate(loader): 
                    model_pd.train()

                    pt = model_pd(Hx_dir={'Hx':hx,'edge_index':cinfo['tr']['edge_index']}, 
                                  B=B, m=M, k=k_weights)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    (-model_pd.lagr).backward()
                    torch.nn.utils.clip_grad_norm_(model_pd.parameters(), 5)
                    optimizer.step()
                    model_pd.update(**update_dict(i, const_c))

                    if not i%10:
                        print(i, f'training ep:{ep}, step:{i}', model_pd.l_p.mean().item(), model_pd.l_d.mean().item())

                model_pd.eval()
                log['val'] = logs(log['val'], model_pd, inputdir_val, lkey=const_c, save=save_path, vio={'c':const_c, 'e':None})
                log['test'] = logs(log['test'], model_pd, inputdir_test, lkey=const_c)

            #save log
            import pickle
            with open(save_path + 'log.pk', 'wb') as f:
                pickle.dump(log, f)
