# -*- coding: utf-8 -*-
# Copyright (c) 2025 Jiatong Li
# All rights reserved.
# 
# This software is the confidential and proprietary information
# of Jiatong Li. You shall not disclose such confidential
# information and shall use it only in accordance with the terms of
# the license agreement.

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file',help='the path of the train file')
    parser.add_argument('--test_file',help='the path of the test file')
    parser.add_argument('--valid_file',help='the path of the valid file', default='')
    parser.add_argument('--Q_matrix',help='the path of the q-matrix', default='')
    parser.add_argument('--save_path',help='the save path of all results')
    parser.add_argument('--n_user', help='the number of students in the entire dataset')
    parser.add_argument('--n_item', help='the number of exercises in the entire dataset')
    parser.add_argument('--n_know', help='the number of knowledge points in the entire dataset')
    parser.add_argument('--user_dim', help='the dimension of user vector', default=64)
    parser.add_argument('--item_dim', help='the dimension of item vector', default=2)
    parser.add_argument('--batch_size', help='the batch size in the training phase', default=256)
    parser.add_argument('--lr', help='the learning rate in the training phase', default=7e-4)
    parser.add_argument('--epoch', help='the training epoch', default=10)
    parser.add_argument('--device', help='the running device. cpu or gpu', default='cpu')
    args = parser.parse_args()
    return args 

if __name__ == '__main__':
    args = parse_args()
    print(args.train_file)
    