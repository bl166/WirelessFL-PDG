"""
UTILITY FUNCTIONS FOR FEDRATED LEARNING FRAMEORK 
"""
import sys, os
sys.path.append('/root/PDGNet')

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import trange,tqdm

import sysmodels as sysm
import baselines as basm
import utils


def init_learning_models(nuser, isize, hsize, osize, lrate=1e-3, 
                         lossfunc=nn.CrossEntropyLoss, optimizer=torch.optim.Adam, initfunc=None, seed=None):
    nets = []
    # initialize the neural network of each user and global
    for user in range(nuser+1):
        net = sysm.patternnet(isize, hsize, osize, 
                         learning_rate = lrate, 
                         criterion     = lossfunc,
                         optimizer     = optimizer,
                         initFunc      = initfunc, 
                         seed          = seed,
                         name_string   = f'net{user}' if user<nuser else 'global')
        nets.append(net)
    lnets, gnet = nets[:-1], nets[-1]
    return lnets, gnet


def train_local_models(qvals, usvec, dnums, bsize, data, lnets, gnet, epochs=1, first_iter=False):
    for user in range(len(usvec)):
        if qvals[user] >= 1: # this user is not chosen
            continue

        # Train each neural network:
        # Since each user is associated with a packet error rate, we randomly choose a value;
        # If this value is larger than the packet error rate, then this user will join this FL iteration.
        if first_iter or np.random.rand(1) > qvals[user]: 
            usvec[user] = 1; # join fl
            if isinstance(data, list):
                data_user = data[user]
            else:
                d_indices = torch.arange(sum(dnums[:user]), sum(dnums[:user+1])) 
                data_user = torch.utils.data.Subset(data, d_indices)
            train_loader  = torch.utils.data.DataLoader(data_user, batch_size = bsize, shuffle = True)

            # Change each user' local FL model to global FL model
            lnets[user].model.load_state_dict(gnet.model.state_dict()) 

            # train local FL model of each user and record weights
            lnets[user] = utils.train(lnets[user], train_loader, epochs=epochs);  

    return lnets, usvec


def aggregate_global_model(usvec, kweights, lnets, gnet):
    # number of users joining the curr iter
    finalb = np.where(usvec)[0]; 
    n_users = len(finalb)
    aflag = False

    # average all users parameters  
    if n_users > 0:                
        state_dict_new = lnets[finalb[0]].model.state_dict()
        for jj,fj in enumerate(finalb):
            state_dict_curr = lnets[fj].model.state_dict()
            for key in state_dict_new:
                if jj == 0:
                    state_dict_new[key] *= kweights[fj] 
                elif jj < n_users-1:  
                    state_dict_new[key] += state_dict_curr[key] * kweights[fj]
                else:
                    state_dict_new[key] /= kweights[finalb].sum()

        #initialize these matirces used for global FL model update
        gnet.model.load_state_dict(state_dict_new) 
        aflag = True
        
        #print(finalb)
        
    return gnet, aflag


def predict_by_global_model(gnet, data, targets, metric, nsamples = 1000):
    test_indices = np.arange(nsamples)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(data, test_indices), batch_size=nsamples, shuffle=False
    )
    if metric == 'err_rate':
        pred = utils.predict(gnet.model, test_loader, 'pred')
        diff = (pred - targets[test_indices]).cpu().numpy()
        err = (diff != 0).mean()
    elif metric == 'rmse':
        pred = utils.predict(gnet.model, test_loader, 'logits')
        diff = (pred - targets[test_indices]).cpu().numpy()
        err = (diff ** 2).mean()**.5
    else:
        raise        
    return err


def power_alloc_per(scheme, nuser, nuser_sel, kweights, wsize, h, beta, pm, B, constr, qvec_ready=None):
    scheme_string = scheme.lower()
    def naive_pa():
        # initial power
        if 'rand' in scheme_string:
            P = pm * np.tile( np.random.rand(1, nuser).T, (1, nuser_sel) ) 
        elif 'orth' in scheme_string:
            P = pm * np.ones((nuser, nuser_sel))
        else:
            raise
            
        # system model
        sysmodel = sysm.WirelessModel(bandwidth=B)
        
        # initial interference
        In = sysmodel.power_interference(beta, P) 
        
        # PER of each user over each RB
        q = sysmodel.packet_error_rate(h, In ,P)
        
        # Uplink data rate of each user over each RB
        SINR   = sysmodel.signal_interf_noise_ratio(h[:nuser], In, P)
        rateu  = sysmodel.data_rate(SINR, B);
        
        # Uplink delay of each user over each RB 
        delay  = wsize / rateu
        
        #Sum energy consumption of each user
        energy = sysmodel.consumed_energy(P, wsize, delay, pm) 

        # drop users violating requirements
        finalq = basm.pa_orthog(q, delay, energy, *constr, 
                           kweights, nuser, nuser_sel)
        return finalq
        
    if 'ideal' in scheme_string:
        return np.zeros(nuser)
    elif 'pd' in scheme_string:
        return qvec_ready
    else:
        return naive_pa()
    
    return finalq
        
        

def FL_main(inputs,
            fl_task,
            allocation_scheme,
            usernumber, 
            availnumber,
            datanumber    = None,
            kweights      = None,
            iteration     = 50, 
            iter_epochs   = 1,
            averagenumber = 5,
            per_ready     = None,
            seed          = None
           ):
    # decompose inputs
    h_all, beta_all, p_max = inputs['channels']
    test_set, train_sets, batch_size_train = inputs['fedtask']
    drequirement, erequirement = inputs['require']
    numberofneuron, in_size, out_size = inputs['configs']
    msize = in_size*numberofneuron + numberofneuron*out_size + (numberofneuron+out_size) #=39760
    Z = msize * 16 / 1024 / 1024
       
    # set some hyper params
    if 'mnist' in fl_task.lower():
        learning_rate  = 1e-3
        loss_function  = nn.CrossEntropyLoss
        optim_function = torch.optim.Adam
        init_func      = utils.initnw
        final_metric   = 'err_rate'
        n_test_samples = 1000        
    elif 'airq' in fl_task.lower():
        learning_rate  = 8e-4#0.015
        loss_function  = nn.MSELoss
        optim_function = torch.optim.Adam#torch.optim.SGD
        init_func      = None
        final_metric   = 'rmse'
        n_test_samples = 100
    else:
        raise
        
    if datanumber is None:
        datanumber =(np.random.RandomState(42).randint(100,1000,usernumber) / 5).astype(int)
    if kweights is None:
        kweights  = datanumber/datanumber.sum()
        
    averageerror = np.zeros(averagenumber)
    testerror = np.ones((averagenumber, iteration)) 
        
    # multiple random runs
    selected_num_workers = np.zeros((averageerror.size, iteration, usernumber))
    for ai in trange(averageerror.size):   
        nets, global_net = init_learning_models(nuser     = usernumber, 
                                                isize     = in_size, 
                                                hsize     = numberofneuron, 
                                                osize     = out_size, 
                                                lrate     = learning_rate, 
                                                lossfunc  = loss_function, 
                                                optimizer = optim_function,
                                                initfunc  = init_func,
                                                seed      = seed+ai
                                               )

        success_mat = np.zeros((iteration,usernumber)); # indicates users successfully transmitting
        error_vec   = np.nan * np.zeros(iteration);   # evaluates global net at each iter
        iterationtime = np.zeros(iteration);

        if seed is None:
            indices = np.arange(iteration)
        else:
            indices = np.random.RandomState(seed+ai).permutation(iteration)
        for fi,ii in enumerate(tqdm(indices, desc = allocation_scheme)):
            i = ii%h_all.shape[0]          

            # PER as a result of power allocation 
            finalq = power_alloc_per(allocation_scheme, 
                                     nuser      = usernumber, 
                                     nuser_sel  = availnumber, 
                                     kweights   = datanumber,
                                     wsize      = Z,
                                     h          = h_all[i].reshape(-1,1), 
                                     beta       = beta_all[i], 
                                     pm         = p_max[i], 
                                     B          = 1,
                                     constr     = (drequirement, erequirement),
                                     qvec_ready = per_ready[i] if per_ready is not None else None
                                    )

            # train local models to be transimitted
            nets, success_mat[fi] = train_local_models(qvals = finalq, 
                                                      usvec = success_mat[fi], 
                                                      dnums = datanumber, 
                                                      bsize = batch_size_train, 
                                                      data  = train_sets, 
                                                      lnets = nets, 
                                                      gnet  = global_net, 
                                                      epochs= iter_epochs,
                                                      first_iter = fi==0
                                                     )
            # calculate the global model
            global_net, aflag = aggregate_global_model(success_mat[fi], datanumber, nets, global_net)


            # calculate the prediction errors at iteration i 
            if aflag:
                error_vec[fi] = predict_by_global_model(gnet     = global_net,
                                                        data     = test_set,
                                                        targets  = test_set.targets,
                                                        metric   = final_metric,
                                                        nsamples = n_test_samples )
            else:
                error_vec[fi] = error_vec[i-1] 
            print(fi, i, error_vec[fi], end=' \r')    

        testerror[ai] = error_vec.reshape(-1)
        print(f'Error at iter#{iteration} is {error_vec.min()}')
        averageerror[ai]=error_vec.min();
        selected_num_workers[ai] = success_mat#.sum(1)

    return testerror, global_net, selected_num_workers
       

    
"""
Non-IID Fed Learning
"""

def data_partition_dir(dataset, dnumbers, num_classes, alpha=0.1, least_samples=10, seed=42):
    # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
   
    num_clients = len(dnumbers)
    num_total = sum(dnumbers)
    
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = dataset.data, dataset.targets

    dataidx_map = {}

    min_size = 0
    K = num_classes
    N = len(dataset_label)

    cnt = 0
    while min_size < least_samples:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = np.where(dataset_label == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.RandomState(seed + cnt).dirichlet(np.repeat(alpha, num_clients))
            cnt += 1
            proportions = np.array([p*(len(idx_j)< N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_clients):
        #dataidx_map[j] = idx_batch[j]
        dataidx_map[j] = np.random.RandomState(seed+j).permutation(idx_batch[j])[:dnumbers[j]]

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))
    return X, y, statistic


def add_ni_noise(dss, dnums, scaler, sdvals=None, seed=0):
    # add normal noise to local workers
    if sdvals is None:
        sdvals = np.arange(1., len(dnums)+1.) 
    for u, sv in enumerate(sdvals):
        if isinstance(dss, list):
            data = dss[u].dataset.data[dss[u].indices].float()
            noise = scaler * sv * torch.normal(
                0, 1, data.shape, generator=torch.manual_seed(seed+u)
            )
            data = torch.clip(data + noise, min = data.min(), max = data.max())
            dss[u].dataset.data[dss[u].indices] = data.to(dss[u].dataset.data.dtype) # don't need to worry about range (uint8)
        else:
            d_idx = torch.arange(sum(dnums[:u]), sum(dnums[:u+1])) 
            data = dss.data[d_idx].float()
            noise = scaler * sv * torch.normal(
                0, 1, data.shape, generator=torch.manual_seed(seed+u)
            ) # normal noise
            data = torch.clip(data + noise, min = data.min(), max = data.max())
            dss.data[d_idx] = data.to(dss.data.dtype)
    return dss

    
def per_from_learning_models(modeldir, numworker, inputdir):
    ## load model
    mfile = modeldir + 'model_pd.pt'
    mfilev = modeldir + 'model_pd-vio.pt'

    if os.path.exists(mfile):
        model = torch.load(mfile)
    else:
        model = torch.load(mfilev)
    model.eval()    
        
    # for compatibility
    model.model.net_size = numworker
    model = gconv_ver_compat_helper(model)
    
    # allocate power
    with torch.no_grad():
        pt = model.allocate(inputdir['Hx_dir'])
    finalq = sysm.f1(pt, inputdir['Hx_dir']['Hx'], inputdir['m'])
    finalq_cond = (1-finalq*(pt>0)).cpu().numpy()   
    return finalq_cond
    
    
def gconv_ver_compat_helper(model):
    # pyg new version compatibility conversion
    if 'GCN' in str(model.model.__class__):
        model.model.sig_size = 1
        for i, gconv in enumerate(model.model.model.hidden):
            if 'lin' not in gconv.__dict__:
                try:
                    w = gconv.weight
                except:
                    break
                gconv.lin = nn.Linear(*w.shape).to(w.device)
                with torch.no_grad():
                    gconv.lin.weight.copy_(w.T)

                gconv._explain = None
                gconv.decomposed_layers = 1

                # Hooks -- https://github.com/pyg-team/pytorch_geometric/blob/bc836242ab9c8c6fec96d5c6733a6df7aeb95802/torch_geometric/nn/conv/message_passing.py#L170
                gconv._propagate_forward_pre_hooks = {}
                gconv._propagate_forward_hooks = {}
                gconv._message_forward_pre_hooks = {}
                gconv._message_forward_hooks = {}
                gconv._aggregate_forward_pre_hooks = {}
                gconv._aggregate_forward_hooks = {}
                gconv._message_and_aggregate_forward_pre_hooks = {}
                gconv._message_and_aggregate_forward_hooks = {}
                gconv._edge_update_forward_pre_hooks = {}
                gconv._edge_update_forward_hooks = {}
    return model
