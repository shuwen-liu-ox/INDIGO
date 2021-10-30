import numpy as np
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils_test import encoding_test
from utils.accuracy import compute_accuracy_for_test
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

acc_list = [] 
precision_list = []
recall_list = []
f1_list = []
false_positive_rate_list = []
false_negative_rate_list = []
roc_auc_list = []
auc_pr_list = []

r_mr_list = []
r_mrr_list = []
r_hits1_list = []
r_hits3_list = []
r_hits10_list = []


for run in range(args.num_runs):

    adj, features, labels, masks, num_type, num_relation, constants, relations, types, pairs, hits_true, r_hits_candidates = encoding_test(run, train_dataset, test_dataset)

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
    if args.print:
        f_test = open("predictions_{}.txt".format(args.dataset), "w+")
        for p_id in range(len(pairs)):
            for r_id in range(num_type + 2 * num_relation ):
                if masks[p_id][r_id] ==1:
                    if r_id < num_type:
                        f_test.write(constants[pairs[p_id][0]])
                        f_test.write("\t")
                        f_test.write("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
                        f_test.write("\t")
                        f_test.write(types[r_id])
                    elif r_id >= num_type and r_id < num_type+num_relation:
                        f_test.write(constants[pairs[p_id][0]])
                        f_test.write("\t")
                        f_test.write(relations[r_id - num_type])
                        f_test.write("\t")
                        f_test.write(constants[pairs[p_id][1]])

                    else:
                        f_test.write(constants[pairs[p_id][1]])
                        f_test.write("\t")
                        f_test.write(relations[r_id - num_type - num_relation])
                        f_test.write("\t")
                        f_test.write(constants[pairs[p_id][0]])
                    f_test.write("\t")
                    f_test.write(str(output_test_accuracy[p_id][r_id].item()))
                    f_test.write("\t")
                    f_test.write(str(labels[p_id][r_id].item()))
                    f_test.write("\n")
        f_test.close()
        print("Predicted triples saved in file predictions_{}.txt".format(args.dataset))
    output_test = torch.mul(output_test, masks)

    loss = nn.BCELoss()
    loss_test = loss(output_test, labels)

    score_threshold = 0.5

    acc_test, precision_test, recall_test, f1_test, false_positive_rate_test, false_negative_rate_test, roc_auc_test, auc_pr_test, r_mr_test, r_mrr_test, r_hits1_test, r_hits3_test, r_hits10_test = compute_accuracy_for_test(output_test_accuracy, labels, masks, score_threshold, num_relation, num_type, hits_true, r_hits_candidates)

    acc_list.append(acc_test.item())
    precision_list.append(precision_test.item())
    recall_list.append(recall_test.item())
    f1_list.append(f1_test.item())
    false_positive_rate_list.append(false_positive_rate_test.item())
    false_negative_rate_list.append(false_negative_rate_test.item())
    roc_auc_list.append(roc_auc_test)
    auc_pr_list.append(auc_pr_test)

    r_mr_list.append(r_mr_test)
    r_mrr_list.append(r_mrr_test)
    r_hits1_list.append(r_hits1_test)
    r_hits3_list.append(r_hits3_test)
    r_hits10_list.append(r_hits10_test)


    
print('------------Classification-------')

print('accuracy: {:.4f}, var:{:.4f}\n'.format(np.mean(acc_list), np.var(acc_list)),
      'precision: {:.4f}, var:{:.4f}\n'.format(np.mean(precision_list), np.var(precision_list)),
      'recall: {:.4f}, var:{:.4f}\n'.format(np.mean(recall_list), np.var(recall_list)),
      'f1: {:.4f}, var:{:.4f}\n'.format(np.mean(f1_list), np.var(f1_list)),
      'false_positive_rate: {:.4f}, var:{:.4f}\n'.format(np.mean(false_positive_rate_list), np.var(false_positive_rate_list)),
      'false_negative_rate: {:.4f}, var:{:.4f}\n'.format(np.mean(false_negative_rate_list), np.var(false_negative_rate_list)),
      'roc_auc: {:.4f}, var:{:.4f}\n'.format(np.mean(roc_auc_list), np.var(roc_auc_list)),
      'auc_pr: {:.4f}, var:{:.4f}\n'.format(np.mean(auc_pr_list), np.var(auc_pr_list)))

print('------------Ranking--------------')

print(
      'r-MR: {:.4f}, var:{:.4f}\n'.format(np.mean(r_mr_list), np.var(r_mr_list)),
      'r-MRR: {:.4f}, var:{:.4f}\n'.format(np.mean(r_mrr_list), np.var(r_mrr_list)),
      'r-HITS@1: {:.4f}, var:{:.4f}\n'.format(np.mean(r_hits1_list), np.var(r_hits1_list)),
      'r-HITS@3: {:.4f}, var:{:.4f}\n'.format(np.mean(r_hits3_list), np.var(r_hits3_list)),
      'r-HITS@10: {:.4f}, var:{:.4f}\n'.format(np.mean(r_hits10_list), np.var(r_hits10_list)),
)      


