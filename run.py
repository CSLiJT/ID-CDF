#! python3
# -*- encoding: utf-8 -*-

from model_parser import parse_args
import gc
import json
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn 
import model 
import train 
import sys
import os
from tools import degree_of_consistency

def add_knowledge_code(data: pd.DataFrame, Q_mat):
    knowledge = []
    for i in range(data.shape[0]):
        knowledge.append(Q_mat[data.loc[i,'item_id']])
    data['knowledge'] = knowledge
    return data 

if __name__ == '__main__':
    args = parse_args()

    df_train = pd.read_csv(args.train_file)
    df_valid = pd.read_csv(args.valid_file)
    df_test = pd.read_csv(args.test_file)

    n_user = int(args.n_user)
    n_item = int(args.n_item)
    n_know = int(args.n_know)

    Q_mat = np.load(args.Q_matrix) if args.Q_matrix !='' else np.ones((n_item, n_know))

    df_train = add_knowledge_code(df_train, Q_mat)
    df_valid = add_knowledge_code(df_valid, Q_mat)
    df_test = add_knowledge_code(df_test, Q_mat)

    itf_type = args.itf_type
    user_dim = int(args.user_dim)
    item_dim = int(args.item_dim)
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    epoch = int(args.epoch)
    eta = float(args.eta)
    device = torch.device(args.device)
    print(device)

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    acc_all=[]
    auc_all = []
    rmse_all=[]
    recall_all=[]
    precision_all=[]
    f1_all = []
    for i in range(1):
        net = model.IDCD(n_user, n_item, n_know, user_dim, \
            item_dim, itf_type, Q_mat = Q_mat, \
            monotonicity_assumption = True,\
            eta_s = eta, eta_e = eta, device=device)
        result_all = train.train(net, df_train, df_valid, batch_size = batch_size, \
            lr = lr, n_epoch = epoch)
        np.save(os.path.join(save_path, 'result_all.npy'), result_all)
        test_result = train.eval(net, df_test,batch_size=256)
        acc_all.append(test_result['acc'])
        auc_all.append(test_result['auc'])
        rmse_all.append(test_result['rmse'])
        recall_all.append(test_result['recall'])
        precision_all.append(test_result['precision'])
        f1_all.append(test_result['f1'])

    print('acc = %.3f ± %.3f'%(np.mean(acc_all),np.std(acc_all)))
    print('auc = %.3f ± %.3f'%(np.mean(auc_all),np.std(auc_all)))
    print('rmse = %.3f ± %.3f'%(np.mean(rmse_all),np.std(rmse_all)))
    print('recall = %.3f ± %.3f'%(np.mean(recall_all),np.std(recall_all)))
    print('precision = %.3f ± %.3f'%(np.mean(precision_all),np.std(precision_all)))
    print('f1 = %.3f ± %.3f'%(np.mean(f1_all),np.std(f1_all)))

    with open(os.path.join(save_path, 'cmd.txt'),'w') as fp:
        fp.write(' '.join(['python']+sys.argv))

    with open(os.path.join(save_path, 'test_result.json'),'w') as fp:
        json.dump(test_result, fp)

    torch.save(net, os.path.join(save_path, 'params_%s_%s_%s.pt'%(itf_type,user_dim,item_dim)))
    gc.collect()
    
