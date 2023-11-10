import os
import gc
import sys
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# SET YOUR PROJECT ROOT DIR HERE
PROJ_RT = os.getcwd()
DATA_RT = os.path.join(PROJ_RT, 'datasets/data/')
MODL_RT = os.path.join(PROJ_RT, 'results/models/')

sys.path.append(PROJ_RT)
sys.path.append(os.path.join(PROJ_RT, 'PDGNet'))
from powernets import MLP_ChPm_PD, GCN_ChPt_PD, WirelessFedL_PrimalDual
from sysmodels import get_utility_func 
from utils import *
from global_vars import *

# CONFIGURE WHETHER TO TRAIN ON GPU
device = torch.device('cuda')
# device = torch.device('cpu')

L    = 8  # 6, 16, 24, 32
M_BS = 1
NR   = 10
DIST = {'tr'  : 'Alessio', 
        'val' : 'Alessio',
        'test': 'Alessio'}
PDB = np.array(range(-40,10+1,1)).tolist()
B = 1
M = 0.023

CONSTRAINTS = 'c+e'
MODEL = GCN_ChPt_PD # MLP_ChPm_PD | GCN_ChPt_PD

if 'MLP' in str(MODEL):
    AFLAG = 'MLP'
    dropout = 0.
    learning_rate = 1e-4
    in_size , out_size = L**2+1 , L
    inner_architect = [
            {'h_sizes': [128, 256, 64, 16, 8], 
             'activs': ['relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid']
            }
    ]
elif 'GCN' in str(MODEL):
    AFLAG = 'GCN'
    dropout = 0.
    learning_rate = 5e-4
    in_size , out_size = 1 , 1
    inner_architect = [
            {'h_sizes': [16, 32, 64, 16, 2], 
             'activs': ['elu', 'elu', 'elu', 'elu', 'elu', 'sigmoid']
            }
    ]
else:
    raise


l2        = 1e-6
epochs    = 1000
save_freq = 100
RSEED     = 42

# save results at
RSLT_PATH = lambda intf,pdbm,constr: \
    MODL_RT+f"/{DIST['tr']}_Ant+{NR}_User+{L}_{AFLAG}_Intfx{intf}_Pmax{pdbm:+}_Constr+{constr:s}_LR{learning_rate:+.1e}_seed+{RSEED}/"


####################################################
####################   DATA   ######################
####################################################

def get_data(i_scale, p_max):
    
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
        X[phase][:,L:-1][:,indexing]*=i_scale
        
        # with intended pmax
        X[phase] = X[phase][PDB.index(p_max)::len(PDB),:]
        y[phase] = X[phase][:,:L]

        cinfo[phase] = dict_to_device(cinfo[phase], device)
        print(phase, X[phase].shape, y[phase].shape, X[phase][0,L])

    return X,y,cinfo
    
    
    
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
    
    for kc, lmbd in model.lambdas.items():
        append_as_dict_vals(logdir[lkey], 'lambda_'+kc, lmbd.cpu().detach().numpy())
    for kc, var in model.vars.items():
        append_as_dict_vals(logdir[lkey], 'var_'+kc, var.cpu().detach().numpy())
        
    vflag = False
    for kc in ['q','c','e']:
        if kc in model.Ef:
            ev = model.Ef[kc].cpu().detach().numpy()
            append_as_dict_vals(logdir[lkey], 'Ef_'+kc, ev)
            
            # set constraint violation indicator
            vflag = False
            if vio is not None:
                if kc in vio:
                    if vio[kc] is not None:
                        try:
                            vflag = np.any(ev < vio[kc])
                        except:
                            vflag = ev < vio[kc]
        else:
            ef = get_utility_func(kc)(pt, inputdir['Hx_dir']['Hx'], inputdir['B'])
            mask = ef>0
            ev = ((ef*mask).sum(dim=0)/mask.sum(dim=0,keepdim=True) ).cpu().detach().numpy()
            append_as_dict_vals(logdir[lkey], 'Ef_'+kc, ev)
        
    if not vflag: 
        # if best performance, and satisfy constraints
        append_as_dict_vals(logdir[lkey], '_l_p_sat', lp)
        if np.all(logdir[lkey]['_l_p_sat'][-1] >= np.array(logdir[lkey]['_l_p_sat'])) and save:
            torch.save(model_pd, save + 'model_pd.pt')
    else:
        if np.all(logdir[lkey]['l_p'][-1] >= np.array(logdir[lkey]['l_p'])) and save:
            torch.save(model_pd, save + 'model_pd-vio.pt')
            
    return logdir


def construct_constr_str(Rc, Ec):
    constr_str = []
    for cc in CONSTRAINTS.split('+'):
        if cc=='c' and Rc is not None:
            constr_str.append(cc+f'{Rc:+.2e}')
        elif cc=='e' and Ec is not None:
            constr_str.append(cc+f'{Ec:+.2e}')
        else:
            raise
    constr_str = '_'.join(constr_str)  
    return constr_str

##
    
for i_scale in [1,2,4,8]:
    if i_scale==1:
        pmax_val_set = [-40, -30, -20, -10, 0]
    else:
        pmax_val_set = [-20]  
        
    for pmax_val in [-20]:#pmax_val_set:# [-20]:#
        Rc = CDICT[I_LOOP.index(i_scale), P_LOOP.index(pmax_val)]
        Ec = EDICT[I_LOOP.index(i_scale), P_LOOP.index(pmax_val)]
        print(f'*Starting* Interference x{i_scale}, Pmax @{pmax_val} index, Rc = {Rc}, Ec = {Ec} ...')
            
        save_path = RSLT_PATH(i_scale, pmax_val,construct_constr_str(Rc, Ec))
        if not os.path.exists(save_path): 
            os.makedirs(save_path)
        save_log_path = save_path + 'log.pk'
        #if os.path.exists(save_log_path): continue

        X, y, cinfo = get_data(i_scale, pmax_val)

        datanumbers = np.random.RandomState(RSEED).randint(100,1000,L).astype('float')
#         datanumbers = np.array([1]*L) # uniform
        k_weights = torch.from_numpy(datanumbers/datanumbers.sum()).float().to(device)

        #
        update_dict   = lambda ii, cc, ee: {'stepsizes': dict(zip(['q','lq','c','lc','e','le'],
                                                              [learning_rate/2**(AFLAG=='MLP')]*6)),
                                        'mins': {'c':cc*B, 'e':ee}, 'kw': k_weights}

        for const_c in [Rc]:
            for const_e in [Ec]:
                # key for log dict
                log_key = str(const_c)+'+'+str(const_e)

                # instantiate model   
                model_alloc = MODEL(num_blocks = 1,  # obsolete: this was for unfolding
                                    num_users  = L,
                                    in_size    = in_size, 
                                    out_size   = out_size, 
                                    **inner_architect[0], 
                                    edge_index = None, #cinfo['tr']['edge_index'], 
                                    dropout    = dropout).to(device)
                
                model_pd = WirelessFedL_PrimalDual(model = model_alloc, 
                                                   users = L, 
                                                   kw    = k_weights, 
                                                   constraints=['q']+CONSTRAINTS.split('+'),
                                                   device=device)
                num_params = count_parameters(model_alloc, 1)    
                optimizer = torch.optim.Adam(model_alloc.parameters(), lr=learning_rate, weight_decay=l2)

                # 
                dataset = TensorDataset(X['tr'], y['tr'])
                loader = DataLoader(dataset, batch_size=100, shuffle=True)

                inputdir_val = {'Hx_dir':{'Hx':X['val'], 'edge_index':cinfo['val']['edge_index']}, 
                               'B':B, 'm':M, 'kw':k_weights}
                inputdir_test = {'Hx_dir':{'Hx':X['test'], 'edge_index':cinfo['test']['edge_index']}, 
                               'B':B, 'm':M, 'kw':k_weights}

                # if trained half way, load model and logs
                save_modchk_path = save_path + 'model_pd-latest.pt'
                if os.path.exists(save_log_path) and os.path.exists(save_modchk_path) :
                    print('loading from ...', save_path)
                    log = pickle.load(open(save_log_path, 'rb'))
                    model_pd = torch.load(save_modchk_path)
#                     model_chk = torch.load(save_modchk_path)
#                     model_pd.model.load_state_dict(model_chk.model.state_dict())
                else:
                    log = {'ep':0, 'val':{}, 'test':{}}
                    log['val'][log_key], log['test'][log_key] = {},{}
                    log['val'] = logs(log['val'], model_pd, inputdir_val, lkey=log_key)
                    log['test'] = logs(log['test'], model_pd, inputdir_test, lkey=log_key)                    
                        
                try:
                    for ep in range(log['ep'],epochs):
                        log['ep'] = ep
                        for i, (hx,_) in enumerate(loader): 
                            
                            if ep==0 and i==0:
                                model_pd.init_prime(Hx_dir={'Hx':hx, 'edge_index':cinfo['tr']['edge_index']}, B=B, m=M)
    
                            model_pd.train()
                            pt = model_pd(Hx_dir={'Hx':hx, 'edge_index':cinfo['tr']['edge_index']}, 
                                          B=B, m=M, kw=k_weights)
                            if torch.any(torch.isnan(pt)):
                                raise
                            
                            # zero the parameter gradients
                            optimizer.zero_grad()
                            (-model_pd.loss_bp).backward()
                            torch.nn.utils.clip_grad_norm_(model_pd.parameters(), 5., error_if_nonfinite=False)
                            optimizer.step()
                            model_pd.update(**update_dict(i, const_c, const_e))
                                                 
                            if not i%10:
                                print(i, f'training ep:{ep}, step:{i}', model_pd.l_p.mean().item(), model_pd.l_d.max().item())

                        model_pd.eval()
                        log['val'] = logs(log['val'], model_pd, inputdir_val, lkey=log_key, save=save_path, 
                                          vio={'c':const_c, 'e':const_e})
                        log['test'] = logs(log['test'], model_pd, inputdir_test, lkey=log_key)

                        if ep==epochs-1 or not (ep+1)%save_freq: 
                            with open(save_log_path, 'wb') as f:
                                print('saving to ...', save_path)
                                pickle.dump(log, f)
                            sp_ckpt = save_modchk_path if ep==epochs-1 else (save_path + f'model_pd.pt-ep{ep:05d}')
                            torch.save(model_pd, sp_ckpt)   

                except (KeyboardInterrupt, SystemExit):
                    #save log
                    with open(save_log_path, 'wb') as f:
                        print('saving to ...', save_path)
                        pickle.dump(log, f)
                        torch.save(model_pd, save_modchk_path)   


