import numpy as np
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from utils.utils_test_e_hits import encoding_test
from GNN.model import GCN


import time


#parse the argument

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, required=True,
                    help='Name of dataset')
parser.add_argument('--model_dir', type=str, required=True,
                    help='Directory name where models are saved')
parser.add_argument('--model_name', type=str, required=True,
                    help='Name of testing model')
parser.add_argument('--num_runs', type=int, default=10,
                    help='Number of testing runs')
parser.add_argument('--hidden', type=int, default=64,
                    help='Dimension of hidden vectors')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (will not been used in testing)')
parser.add_argument('--print', action='store_true',
                    help='To print the predicted triples to a file')
args = parser.parse_args()


#encoding the data

train_dataset = args.dataset
test_dataset = args.dataset



e_mr_list = []
e_mrr_list = []
e_hits1_list = []
e_hits3_list = []
e_hits10_list = []

for run in range(args.num_runs):
    return_list = encoding_test(train_dataset, test_dataset)

    ranks_sub = []
    ranks_obj = []


    for r in return_list:
        adj, features, labels, masks, num_type, num_relation, constants, relations, types, pairs, positive_flag, hits_true, e_hits_sub_candidates, e_hits_obj_candidates = r


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

        output_test_accuracy = output_test.clone()




        (true_pair_idx, true_dim_idx) = hits_true
        score = output_test_accuracy[true_pair_idx, true_dim_idx]

        if positive_flag:
            scores = []
            scores.append(score)
            for c in e_hits_sub_candidates:
                pair_false_idx = c[0]
                dim_false_idx = c[1]
                scores.append(output_test_accuracy[pair_false_idx][dim_false_idx].item()) 
                rank_sub = len(scores) - np.argwhere(np.argsort(scores) == 0)
            ranks_sub.append(rank_sub)

            scores = []
            scores.append(score)
            for c in e_hits_obj_candidates:
                pair_false_idx = c[0]
                dim_false_idx = c[1]
                scores.append(output_test_accuracy[pair_false_idx][dim_false_idx].item()) 
                rank_obj = len(scores) - np.argwhere(np.argsort(scores) == 0)
            ranks_obj.append(rank_obj)

    def ranking(ranks):
        mr = np.mean(ranks)
        mrr = np.mean(1 / np.array(ranks))

        isHit1List = [x for x in ranks if x <= 1]
        isHit3List = [x for x in ranks if x <= 3]
        isHit10List = [x for x in ranks if x <= 10]
        hits_1 = len(isHit1List) / len(ranks)
        hits_3 = float(len(isHit3List)) / float(len(ranks))
        hits_10 = len(isHit10List) / len(ranks)
        return mr, mrr, hits_1, hits_3, hits_10

    mr_sub, mrr_sub, hits1_sub, hits3_sub, hits10_sub = ranking(ranks_sub)
    mr_obj, mrr_obj, hits1_obj, hits3_obj, hits10_obj = ranking(ranks_obj)


    e_mr_list.append(np.mean([mr_sub, mr_obj]))
    e_mrr_list.append(np.mean([mrr_sub, mrr_obj]))
    e_hits1_list.append(np.mean([hits1_sub, hits1_obj]))
    e_hits3_list.append(np.mean([hits3_sub, hits3_obj]))
    e_hits10_list.append(np.mean([hits10_sub, hits10_obj]))




print('-------Ranking--------------')

print('e-MR: {:.4f}, var:{:.4f}\n'.format(np.mean(e_mr_list), np.var(e_mr_list)),
  'e-MRR: {:.4f}, var:{:.4f}\n'.format(np.mean(e_mrr_list), np.var(e_mrr_list)),
  'e-HITS@1: {:.4f}, var:{:.4f}\n'.format(np.mean(e_hits1_list), np.var(e_hits1_list)),
  'e-HITS@3: {:.4f}, var:{:.4f}\n'.format(np.mean(e_hits3_list), np.var(e_hits3_list)),
  'e-HITS@10: {:.4f}, var:{:.4f}\n'.format(np.mean(e_hits10_list), np.var(e_hits10_list)))      


