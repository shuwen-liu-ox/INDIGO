import numpy as np
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils_test_pattern import encoding_test
from GNN.model import GCN

import time


#parse the argument

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, required=True,
                    help='Name of dataset')
parser.add_argument('--pattern', type=str, default='pattern1',
                    help='Name of pattern')
parser.add_argument('--model_dir', type=str, required=True,
                    help='Directory name where models are saved')
parser.add_argument('--model_name', type=str, required=True,
                    help='Name of testing model')
parser.add_argument('--hidden', type=int, default=64,
                    help='Dimension of hidden vectors')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (will not been used in testing)')

args = parser.parse_args()


#encoding the data

train_dataset = args.dataset





confidence_list = ['1', '0.9-1', '0.8-0.9', '0.7-0.8']

for c in confidence_list:
    pred_list = []
    count_rules = 0
    path = 'data/rule/{}/{}/confidence{}/'.format(args.dataset, args.pattern, c)
    if not os.path.exists(path):
        continue
    for f in os.listdir(path):
        if f.startswith('test-graph-'):

            test_graph_path = '{}{}'.format(path, f)
            test_fact_path = test_graph_path.replace('graph','fact')



            adj, features, labels, masks, num_type, num_relation, constants, relations, types, pairs, hits_true = encoding_test(test_graph_path, test_fact_path, train_dataset)

            #define the Model
            model = GCN(nfeat=features.shape[1],
                        nhid = args.hidden,
                        nclass=labels.shape[1],
                        dropout=args.dropout)

            model_path = "models/{}".format(args.model_dir)

            #load the model
            model.load_state_dict(torch.load("{}/{}.pkl".format(model_path, args.model_name)))


            """test the model"""

            model.eval()

            output_test = model(features, adj)

            pre = []
            pre_flag = 1
            for t in hits_true:
                score = output_test[t[0]][t[1]].item()
                if score >= 0.5:
                    pre_flag = 1
                else:
                    pre_flag = 0
                    break

            pred_list.append(pre_flag)
            if pre_flag == 1:
                count_rules += 1
    print('Number of rules captured for confidence {}: {}'.format(c, count_rules))
    print('percentage of rules captured for confidence {}: {}'.format(c, np.mean(pred_list)))
    
