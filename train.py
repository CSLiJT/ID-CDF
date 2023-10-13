import gc
import math
import numpy as np 
import pandas as pd 
from sklearn.metrics import \
    accuracy_score, f1_score,\
    mean_squared_error, 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

torch.set_default_tensor_type(torch.FloatTensor)

class IDCDataset(Dataset):
    '''
    the dataset of IDCD. 
    '''
    def __init__(self, df_log: pd.DataFrame, n_user:int, n_item:int, Q_mat = None):
        self.df_log = df_log
        self.log_mat = np.zeros((n_user, n_item))
        self.user_id = df_log['user_id'].values
        self.item_id = df_log['item_id'].values
        self.score = df_log['score'].values
        pbar = tqdm(total = df_log.shape[0],desc='Loading data')
        for i, row in df_log.iterrows():
            self.log_mat[int(row['user_id']), int(row['item_id'])] \
                = (row['score'] - 0.5) * 2
            pbar.update(1)
        pbar.close()
    
    def __getitem__(self, index):
        user_id = self.user_id[index]
        item_id = self.item_id[index]
        return torch.Tensor(self.log_mat[user_id,:]), \
            torch.Tensor(self.log_mat[:, item_id]), \
            torch.LongTensor([user_id]), \
            torch.LongTensor([item_id]), \
            torch.FloatTensor([self.score[index]]) \
    
    def __len__(self):
        return self.user_id.shape[0]

def train(model:IDCD, train_data: pd.DataFrame, valid_data: pd.DataFrame, \
    batch_size, lr, n_epoch):
    model.train()
    device = model.device
    dataset = IDCDataset(train_data, model.n_user, model.n_item)
    dataloader = DataLoader(dataset = dataset, batch_size = batch_size, \
        shuffle = True)
    print(model.Theta_buf.is_leaf, model.Psi_buf.is_leaf)
    optimizer = torch.optim.Adam([
        {'params':model.parameters()}], lr=lr)
    result_per_epoch = []
    for epoch in range(n_epoch):
        result_epoch = {}
        pbar = tqdm(total = len(dataloader),desc = 'Epoch %d'%epoch)
        score_all = []
        pred_all = []
        Theta_old = model.get_Theta_buf().numpy().copy()
        for i, (user_log, item_log, user_id, item_id, score) \
            in enumerate(dataloader):

            user_log = user_log.to(device)
            item_log = item_log.to(device)
            user_id = user_id.to(device)
            item_id = item_id.to(device)
            score = score.to(device)
            pred = model(user_log, item_log, user_id, item_id)
            loss = F.binary_cross_entropy(pred, score)
            score_all += score.detach().cpu().numpy().reshape(-1,).tolist()
            pred_all += pred.detach().cpu().numpy().reshape(-1,).tolist()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            optimizer.step()
            pbar.update(1)
        pbar.close()

        model.eval()

        # Update examinee traits
        for i in range(math.ceil(dataset.log_mat.shape[0]/batch_size)):
            idx = np.arange(i*batch_size, min(dataset.log_mat.shape[0]\
                , (i+1)*batch_size))
            model.update_Theta_buf(model.diagnose_theta(\
                torch.Tensor(dataset.log_mat[idx,:])\
                .to(device)).detach(),torch.LongTensor(idx))

        # Update question features
        for i in range(math.ceil(dataset.log_mat.shape[1]/batch_size)):
            idx = np.arange(i*batch_size, min(dataset.log_mat.shape[1]\
                ,(i+1)*batch_size))
            model.update_Psi_buf(model.diagnose_psi(\
                torch.Tensor(dataset.log_mat[:,idx].T)\
                .to(device)).detach(),torch.LongTensor(idx))
        model.train()
        Theta_new = model.get_Theta_buf().numpy().copy()

        score_all = np.array(score_all)
        pred_all = np.array(pred_all)
        
        Theta_norm = np.sqrt(np.sum(np.abs(Theta_new-Theta_old)))
        acc = accuracy_score(score_all, pred_all > 0.5)
        print('Theta_old.head =',Theta_old[:5,0])
        print('Theta_new.head =',Theta_new[:5,0])
        print('epoch = %d, theta_norm = %.6f, acc = %.6f'%\
            (epoch, Theta_norm, acc))
        result_epoch['Theta_old_head'] = Theta_old[:5,:5]
        result_epoch['Theta_new_head'] = Theta_new[:5,:5]
        result_epoch['Theta_norm'] = Theta_norm 
        result_epoch['train_eval'] = {'acc':acc}

        if valid_data is not None:
            result_epoch['valid_eval'] = eval(model, valid_data, batch_size=16)
        result_per_epoch.append(result_epoch)
    return result_per_epoch

def get_eval_result(s_true, s_pred, s_pred_label):
    acc = accuracy_score(s_true, s_pred_label)
    f1 = f1_score(s_true, s_pred_label)
    rmse = np.sqrt(mean_squared_error(s_true, s_pred))
    print('acc = %.6f f1 = %.6f rmse = %.6f'%(acc,f1,rmse))
    return{'acc': acc, 'f1': f1, 'rmse': rmse}


def eval(model:IDCD, data: pd.DataFrame, batch_size):
    model.eval()
    device = model.device
    eval_result = {}
    dataset = IDCDataset(data, model.n_user, model.n_item)
    dataloader = DataLoader(dataset = dataset, \
        batch_size = batch_size, shuffle = False)
    y_pred = []
    for i, (user_log, item_log, user_id, item_id, score) \
        in enumerate(dataloader):

        user_log = user_log.to(device)
        item_log = item_log.to(device)
        user_id = user_id.to(device)
        item_id = item_id.to(device)
        pred_1_batch = model.forward_using_buf(user_id, \
            item_id).detach().cpu().tolist()
        y_pred += pred_1_batch
    y_pred = np.array(y_pred)
    y_true = data['score'].values.astype(int)
    y_plab = (y_pred > 0.5).astype(int)
    eval_result = get_eval_result(y_true, y_pred, y_plab)
    return eval_result
