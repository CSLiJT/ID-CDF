import os
from xml.etree.ElementTree import TreeBuilder
import numpy as np
from numpy.lib.arraysetops import unique
import pandas as pd
import time
import torch
from datetime import datetime
from tqdm import tqdm

class Logger:
    '''
    logger suitable for anywhere
    '''
    def __init__(self, path = './log/', mode='both'):
        self.fmt = "%Y-%m-%d-%H:%M:%S"
        self.begin_time = time.strftime(self.fmt,time.localtime())
        self.path = os.path.join(path,self.begin_time +'/')

    
    def write(self, message: str, mode = 'both'):
        '''
        @Param mode: 
            file(default): print to file
            console: print to screen
            both: print to both file and screen
        '''
        current_time = time.strftime(self.fmt,time.localtime())
        begin = datetime.strptime(self.begin_time,self.fmt)
        end = datetime.strptime(current_time, self.fmt)
        minutes = (end - begin).seconds
        record = '{} ({} s used) {}\n'.format(current_time , minutes, message)

        if mode == 'file' or mode == 'both':
            if not os.path.exists(self.path):
                os.makedirs(self.path)

        if mode == 'file':
            with open(self.path+'log.txt','a') as f:
                f.write(record)

        elif mode == 'console':
            print(record, end='')
        
        elif mode == 'both':
            with open(self.path+'log.txt','a') as f:
                f.write(record)
            print(record, end='')
        
        else:
            print('Logger error! [mode] must be \'file\' or \'console\' or \'both\'.')

def labelize(y_pred: torch.DoubleTensor, threshold = 0.5)->np.ndarray:
    return (y_pred > threshold).to('cpu').detach().numpy().astype(np.int).reshape(-1,)

def to_numpy(y_pred: torch.DoubleTensor)->np.ndarray:
    return y_pred.to('cpu').detach().numpy().reshape(-1,)

def degree_of_consistency(theta_mat: np.array, user_know_hit: np.array, \
    log_mat: np.array, Q_mat: np.array, know_list = None):
    '''
    theta_mat: (n_user, n_know): the diagnostic result matrix
    user_know_hit: (n_user, n_know): the (i,j) element indicate \
        the number of hits of the i-th user on the j-th attribute
    log_mat: (n_user, n_exer): the matrix indicating whether the \
        student has correctly answered the exercise (+1) or not(-1) 
    Q_mat: (n_exer, n_know)
    '''
    n_user, n_know = theta_mat.shape 
    n_exer = log_mat.shape[1]
    doa_all = []
    know_list = list(range(n_know)) if know_list is None else know_list
    for know_id in know_list:
        Z = 1e-9
        dm = 0
        exer_list = np.where(Q_mat[:,know_id] > 0)[0]
        user_list = np.where(user_know_hit[:,know_id]>0)[0]
        n_u_k = len(user_list)
        pbar = tqdm(total = n_u_k * (n_u_k - 1), desc='know_id = %d'%know_id)
        for a in user_list:
            for b in user_list:
                # if m_ak != m_bk, then either m_ak > m_bk or m_bk > m_ak
                if a == b:
                    continue
                Z += (theta_mat[a, know_id] > theta_mat[b, know_id])
                nab = 1e-9
                dab = 1e-9
                for exer_id in exer_list:
                    Jab = (log_mat[a,exer_id] * log_mat[b,exer_id] != 0)
                    nab += Jab * (log_mat[a, exer_id] > log_mat[b, exer_id])
                    dab += Jab * (log_mat[a, exer_id] != log_mat[b, exer_id])
                dm += (theta_mat[a, know_id] > theta_mat[b, know_id]) * nab / dab 
                pbar.update(1)

        doa = dm / Z 
        doa_all.append(doa)
    return doa_all
                