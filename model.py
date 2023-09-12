from collections import OrderedDict
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn 
import torch.nn.functional as F

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)
        
class IDCD(nn.Module):
    def __init__(self, n_user: int, n_item: int, n_know: int, \
        user_dim: int, item_dim: int, \
        Q_mat: np.array = None, \
        monotonicity_assumption: bool = False,\
        device = torch.device('cpu')):
        super(IDCD,self).__init__()
        self.n_user = n_user 
        self.n_item = n_item 
        self.n_know = n_know
        self.user_dim = user_dim 
        self.item_dim = item_dim
        self.itf = self.ncd_func
        self.device = device

        self.Q_mat = torch.Tensor(Q_mat) \
            if Q_mat is not None else torch.ones((n_item, n_know))

        self.K_diff_mat = nn.Parameter(torch.zeros((n_know, user_dim)),\
            requires_grad=False).to(device)
        self.K_diff_mat.requires_grad = True

        self.Q_mat = self.Q_mat.to(device)

        # Buffer of examinee traits
        self.Theta_buf = nn.Parameter(torch.zeros((n_user, n_know))\
            , requires_grad=False).to(device)

        # Buffer of question feature traits
        self.Psi_buf = nn.Parameter(torch.zeros((n_item, n_know))\
            , requires_grad=False).to(device)
        
        f_linear = nn.Linear if monotonicity_assumption is False else PosLinear


        self.f_nn = nn.Sequential(
            OrderedDict(
                [
                    ('f_layer_1', f_linear(n_item, 256)),
                    ('f_activate_1', nn.Sigmoid()),
                    ('f_layer_2', f_linear(256, n_know)),
                    ('f_activate_2', nn.Sigmoid())
                ]
            )
        ).to(device)

        self.g_nn = nn.Sequential(
            OrderedDict(
                [
                    ('g_layer_1', nn.Linear(n_user, 512)),
                    ('g_activate_1', nn.Sigmoid()),
                    ('g_layer_2', nn.Linear(512, 256)),
                    ('g_activate_2', nn.Sigmoid()),
                    ('g_layer_3', nn.Linear(256, n_know)),
                    ('g_activate_3', nn.Sigmoid())
                ]
            )
        ).to(device)

        self.theta_agg_mat = f_linear(n_know, user_dim).to(device)
        self.psi_agg_mat = nn.Linear(n_know, item_dim).to(device)

        self.ncd = nn.Sequential(
            OrderedDict([
                ('pred_layer_1', nn.Linear(user_dim, 64)),
                ('pred_activate_1', nn.Sigmoid()),
                ('pred_dropout_1', nn.Dropout(p=0.5)),
                ('pred_layer_2', nn.Linear(64, 32)),
                ('pred_activate_2', nn.Sigmoid()),
                ('pred_dropout_2', nn.Dropout(p=0.5)),
                ('pred_layer_3', nn.Linear(32, 1)),
                ('pred_activate_3', nn.Sigmoid()),

            ])
        ).to(device)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def ncd_func(self, theta, psi):
        assert(self.user_dim == self.item_dim)
        y_pred = self.ncd(theta - psi)
        return y_pred

    def diagnose_theta(self, user_log: torch.Tensor):
        theta = self.f_nn(user_log)
        return theta

    def diagnose_psi(self, item_log: torch.Tensor):
        psi = self.g_nn(item_log)
        return psi

    def diagnose_theta_psi(self,  user_log: torch.Tensor, item_log: torch.Tensor):
        theta = self.diagnose_theta(user_log)
        psi = self.diagnose_psi(item_log)
        return theta, psi
    
    def update_Theta_buf(self, theta_new, user_id):
        self.Theta_buf[user_id] = theta_new
    
    def update_Psi_buf(self, psi_new, item_id):
        self.Psi_buf[item_id] = psi_new

    def predict_response(self, theta, psi, Q_batch):
        theta_agg = self.theta_agg_mat(theta * Q_batch)
        psi_agg = self.psi_agg_mat(psi * Q_batch)
        output = self.itf(theta_agg, psi_agg)
        return output

    def forward(self, user_log: torch.Tensor, item_log: torch.Tensor, \
        user_id: torch.LongTensor, item_id: torch.LongTensor):
        theta, psi = self.diagnose_theta_psi(user_log, item_log)
        Q_batch = self.Q_mat[item_id].squeeze(dim=1)
        output = self.predict_response(theta, psi, Q_batch)
        return output

    def forward_using_buf(self, user_id: torch.LongTensor, \
        item_id: torch.LongTensor):
        theta = self.Theta_buf[user_id].squeeze(dim=1)
        psi = self.Psi_buf[item_id].squeeze(dim=1)
        Q_batch = self.Q_mat[item_id].squeeze(dim=1)
        output = self.predict_response(theta, psi, Q_batch)
        return output

    def get_Theta_buf(self):
        return self.Theta_buf.detach().cpu()

    def get_Psi_buf(self):
        return self.Psi_buf.detach().cpu()