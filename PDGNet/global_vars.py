import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import trange

import os
import gc
import sys
import h5py
import shutil
import pickle
import numpy as np
import itertools as it
from tqdm.auto import trange,tqdm
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42 
matplotlib.rcParams['lines.markersize'] = 5
BIGGER_SIZE, NORMAL_SIZE = 14, 12
plt.rc('font', size=BIGGER_SIZE, family='serif')          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=NORMAL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=NORMAL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=NORMAL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title
colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
markercycle = it.cycle(('o','v','s','^','D','*','+','x')) 
linestycycle = ['-','--','-.',':']

import sys
sys.path.append('/root/PDGNet')


# SET YOUR PROJECT ROOT DIR HERE
PROJ_RT = os.getcwd()
# DATA_RT = PROJ_RT+'/datasets/data/' 
# MODL_RT = PROJ_RT+'/results/models-nr10-uniform/'

PMIN_COND = 1e-12
PDB = np.array(range(-40,10+1,1))

# constants for the channel system model
CONST_CHNL_SYSMOD = {
    'B'  : 180e3,
    'N0' : 1e-3 * 10**(-174/10),
    'F'  : 10**(3/10),
}
CONST_CHNL_SYSMOD['NOISE'] = CONST_CHNL_SYSMOD['B']*CONST_CHNL_SYSMOD['N0']*CONST_CHNL_SYSMOD['F']

# constants for the fed system model
CONST_FL_SYSMOD = {
    'B' : 1,
    'm' : 0.023,
}

## constraint values 
I_LOOP =  [1, 2, 4, 8]                      # interference scaling: {[1], 2, 4, 8}
P_LOOP =  [-40, -30, -25, -20, -10, 0, 10]  # pmax from {-40, -30, [-20], -10, 0, 10}

CDICT = np.full((len(I_LOOP),len(P_LOOP)), np.nan)
CDICT[I_LOOP.index(1),:] = [0.32, 0.55, 0.65, 0.7, 0.7, 0.7, 0.7]
CDICT[:,P_LOOP.index(-20)] = [0.7, 0.45, 0.35, 0.25]#[0.7, 0.5, 0.3, 0.1]

EDICT = np.full((len(I_LOOP),len(P_LOOP)), np.nan)
# EDICT[I_LOOP.index(1),:] = [32, 54, 55, 55, 55, 55, 55]
EDICT[I_LOOP.index(1),:] = [32, 50, 55, 55, 55, 55, 55]
EDICT[:,P_LOOP.index(-20)] = [55, 40, 30, 20]


