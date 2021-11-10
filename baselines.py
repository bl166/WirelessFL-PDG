## Baseline algorithm

## Refs:
# [1] Chen, Mingzhe, et al. "A joint learning and communications framework for federated learning over wireless networks." IEEE Transactions on Wireless Communications 20.1 (2020): 269-283.
# Implementation adopted from https://github.com/mzchen0/Wireless-FL

import torch
import numpy as np
import warnings
from scipy.optimize import linear_sum_assignment
from utils import decouple_input, scale_to_range


def pa_orthog(q_rate, total_delay, total_energy, d_requirement, e_requirement, data_number, user_number, RB_number):

    # Set value for each adge according to our equation (24) from [1]
    W = np.array(data_number[:q_rate.shape[0]]).reshape(-1,1) * (q_rate-1)
    W *= (total_delay<d_requirement)&(total_energy<e_requirement)
    
    def munkres(costmat):
        nRows, nCols = costmat.shape
        costmat_pad = np.zeros([max(nRows, nCols)]*2)
        costmat_pad[:nRows, :nCols] = costmat

        row_ind, col_ind = linear_sum_assignment(costmat_pad)

        vIdx = col_ind<nCols
        cost = np.trace(costmat[row_ind[vIdx]][:,col_ind[vIdx]])

        assignment = -np.ones(nRows).astype(int)
        assignment[row_ind[vIdx]] = col_ind[vIdx]

        return assignment, cost

    #Use Hungarian algorithm to find the optimal RB allocation
    assignment,result = munkres(W);

    # Calculate final packet error rate of each user
    vInd = assignment>=0
    finalq=np.ones(user_number);
    finalq[vInd] = q_rate[vInd, assignment[vInd]]
    
    return finalq
 
    

