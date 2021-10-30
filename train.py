import numpy as np
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils_train import encoding_train
from utils.accuracy import compute_accuracy_for_train
from GNN.model import GCN


import time


#parse the argument

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, required=True,
                    help='Name of training dataset')
parser.add_argument('--epoch', type=int, default=1000,
                    help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate of the optimizer')
parser.add_argument('--weight_decay', type=float, default=5e-8,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Dimension of hidden vectors')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate')

args = parser.parse_args()

np.random.seed(1)
torch.manual_seed(1)

#encoding the data
adj, features, labels_train, labels_valid, masks_train, masks_valid, num_type, num_relation = encoding_train(args.dataset)

#define the Model
model = GCN(nfeat=features.shape[1],
            nhid = args.hidden,
            nclass=labels_train.shape[1],
            dropout=args.dropout)

#set optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

version = "lr{}_wd{}_hidden{}".format(str(args.lr), str(args.weight_decay), str(args.hidden))



f1_highest = 0
epoch_best = 0
score_threshold = 0.5

def train(epoch):

    t = time.time()
    model.train()
    
    """train"""
    
    output_train = model(features, adj) 
    output_train_accuracy = output_train.clone()
    
    #only compute loss for the positive & negative facts
    output_train = torch.mul(output_train, masks_train)
    
    #standard binary cross entropy loss
    loss = nn.BCELoss()
    loss_train = loss(output_train, labels_train)
    loss_train.backward()
    optimizer.step()
    
    acc_train, precision_train, recall_train, f1_train = compute_accuracy_for_train(output_train_accuracy, labels_train, masks_train, score_threshold, num_type, num_relation)

    """validation"""

    model.eval()
    output_valid = model(features, adj)
    output_valid_accuracy = output_valid.clone()
    
    #only compute loss for the positive & negative facts
    output_valid = torch.mul(output_valid, masks_valid)
    
    loss_valid = loss(output_valid, labels_valid)
    
    acc_valid, precision_valid, recall_valid, f1_valid = compute_accuracy_for_train(output_valid_accuracy, labels_valid, masks_valid, score_threshold, num_type, num_relation)
    
    #report the best epoch according to f1_valid
    global f1_highest
    global epoch_best
    
    if f1_valid  > f1_highest:
        f1_highest = f1_valid 
        epoch_best = epoch
        
    print('Epoch: {:04d}'.format(epoch),
      'loss_train: {:.4f}'.format(loss_train.item()*1000),
      'precision_train: {:.4f}'.format(precision_train.item()),
      'recall_train: {:.4f}'.format(recall_train.item()),
      'F1_train: {:.4f}'.format(f1_train.item()),
      'loss_val: {:.4f}'.format(loss_valid.item()*1000),
      'precision_val: {:.4f}'.format(precision_valid.item()),
      'recall_val: {:.4f}'.format(recall_valid.item()),
      'F1_val: {:.4f}'.format(f1_valid.item()),
      'time: {:.4f}s'.format(time.time() - t))
    
    save_path = "models/{}".format(args.dataset)
    save_model_name = '{}_e{}.pkl'.format(version, epoch)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #save the model
    torch.save(model.state_dict(), '{}/{}'.format(save_path, save_model_name))
    


print("Started training model......")

# train model
t_train_start = time.time()

for epoch in range(args.epoch):
    train(epoch)
    
print("Finished training.")
print("Total time for training: {:.4f}s".format(time.time() - t_train_start))
print("Best epoch:", epoch_best)
