import numpy as np
import scipy.sparse as sp
import torch
import time
import random

from utils.tool import read_data, write_dic, dictionary, normalize, sparse_mx_to_torch_sparse_tensor, split_known
    
    
def encoding_train(dataset = "fb237_v1"):
    
    """load train/validation data, and do the encoding"""
     
    
    print("Start to encoding train/valid dataset for {}".format(dataset))
    
    t_start = time.time()
    
    path = "data"
   
    
    train_path = "{}/{}/train/train-labeled.txt".format(path, dataset)
    valid_path = "{}/{}/train/valid-labeled.txt".format(path, dataset)
    
    constant_dic_path = "{}/{}/train/constant-dic.txt".format(path, dataset)
    relation_dic_path = "{}/{}/train/relation-dic.txt".format(path, dataset)
    type_dic_path = "{}/{}/train/type-dic.txt".format(path, dataset)
    pair_dic_path = "{}/{}/train/pair-dic.txt".format(path, dataset)
    
    train_triples_with_label = read_data(train_path)
    valid_triples_with_label = read_data(valid_path)
    
    all_triples_with_label = train_triples_with_label + valid_triples_with_label

    constant_set = set()
    relation_set = set()
    type_set = set()

    all_real_triples_with_label = []
    all_type_triples_with_label = []
    for triple in all_triples_with_label:
        if triple[1] != "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
            constant_set.add(triple[0])
            constant_set.add(triple[2])
            relation_set.add(triple[1])
            all_real_triples_with_label.append(triple)
        else:
            constant_set.add(triple[0])
            type_set.add(triple[2])
            all_type_triples_with_label.append(triple)
    
    train_real_triples_with_label = []
    train_type_triples_with_label = []
    for triple in train_triples_with_label:
        if triple[1] != "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
            train_real_triples_with_label.append(triple)
        else:
            train_type_triples_with_label.append(triple)


    valid_real_triples_with_label = []
    valid_type_triples_with_label = []
    for triple in valid_triples_with_label:
        if triple[1] != "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
            valid_real_triples_with_label.append(triple)
        else:
            valid_type_triples_with_label.append(triple)
    
    constants = list(constant_set)
    relations = list(relation_set)
    types = list(type_set)
    constant2index = dictionary(constants)
    relation2index = dictionary(relations)
    type2index = dictionary(types)
    
    #generate list of pairs for encoding
    
    pairs = []
    pair_set = set()
    for triple in all_real_triples_with_label:
        sub_idx = constant2index[triple[0]]
        obj_idx = constant2index[triple[2]]
        if sub_idx < obj_idx:
            if (sub_idx, obj_idx) not in pair_set:
                pair_set.add((sub_idx, obj_idx))
                pairs.append((sub_idx, obj_idx))
        if sub_idx > obj_idx:
            if (obj_idx, sub_idx) not in pair_set:
                pair_set.add((obj_idx, sub_idx))
                pairs.append((obj_idx, sub_idx))
    
    for constant_idx in range(len(constants)):
        pairs.append((constant_idx, constant_idx))
    pair2index = dictionary(pairs)
    
    #save the dictionaries as files in the data directory
    
    write_dic(constant_dic_path, constants)
    write_dic(relation_dic_path, relations)
    write_dic(type_dic_path, types)
    write_dic(pair_dic_path, pairs)
    
    print("Num of triples:{}, num of pairs:{}, num of constants:{}, num of relations:{}, num of types:{}".format(len(all_triples_with_label), len(pairs), len(constants), len(relations),len(types)))
    
    print("Start to encode the graph")
    s_time = time.time()
    
    #collect related pairs for each constant
    
    pairs_for_constant = dict([(i,set()) for i in range(len(constants))])
    p_idx = 0
    for pair in pairs:
        p_idx = pair2index[pair]
        c1 = pair[0]
        c2 = pair[1]
        pairs_for_constant[c1].add(p_idx)
        pairs_for_constant[c2].add(p_idx)
    
    #collect neighbors for each pair node
    
    pneighbors_for_pair = dict([(i,set()) for i in range(len(pairs))])
    for c_idx in range(len(constants)):
        pairs_c = set(pairs_for_constant[c_idx])
        #pair and n_pair would contain one common constant
        for pair in pairs_c:
            for n_pair in pairs_c:
                if pair != n_pair:
                    pneighbors_for_pair[pair].add(n_pair)     
    
    #generate edge list
    
    edges = []

    for i in range(len(pairs)):
        pneighbors = pneighbors_for_pair[i]
        for pneighbor in pneighbors:
            edges.append([i, pneighbor])
            edges.append([pneighbor, i])
            
    print("Finished generating edges", time.time() - s_time)
    
    #generate a normalized adjencency matrix (strategy for GCN)
    
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(len(pairs), len(pairs)), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    print("Total time for adj: {:.4f}s".format(time.time() - s_time))
    
    print("Start to generate features, labels, and masks")

    
    def initialize(train_real_triples_with_label, train_type_triples_with_label, valid_real_triples_with_label, valid_type_triples_with_label):
        
        #split training and validation data with a ratio of 9:1 to train in a way like DAE
        
        train_known_real_triples_with_label, train_unknown_real_triples_with_label = split_known(train_real_triples_with_label)
        train_known_type_triples_with_label, train_unknown_type_triples_with_label = split_known(train_type_triples_with_label)
        valid_known_real_triples_with_label, valid_unknown_real_triples_with_label, = split_known(valid_real_triples_with_label)
        valid_known_type_triples_with_label, valid_unknown_type_triples_with_label, = split_known(valid_type_triples_with_label)

        all_known_type_triples_with_label = train_known_type_triples_with_label + valid_known_type_triples_with_label
        all_known_real_triples_with_label = train_known_real_triples_with_label + valid_known_real_triples_with_label

        labels_train = torch.zeros(len(pairs), len(types) + 2*len(relations))
        labels_valid = torch.zeros(len(pairs), len(types) + 2*len(relations))
        masks_train = torch.zeros(len(pairs),  len(types) + 2*len(relations))
        masks_valid = torch.zeros(len(pairs),  len(types) + 2*len(relations))
        features = torch.zeros(len(pairs),  len(types) + 2*len(relations))
        
        #labels and masks are generated for all triples in train/valid set (pos&neg)
        
        for triple in train_type_triples_with_label:
            cons = triple[0]
            typ = triple[2]
            label = triple[3]
            pair_idx= pair2index[(constant2index[cons], constant2index[cons])]
            typ_idx = type2index[typ]
            if label == "1":
                labels_train[pair_idx][typ_idx] = 1
            elif label == "0":
                labels_train[pair_idx][typ_idx] = 0
            masks_train[pair_idx][typ_idx] = 1

        for triple in train_real_triples_with_label:
            sub = triple[0]
            rel = triple[1]
            obj = triple[2]
            label = triple[3]
            sub_idx = constant2index[sub]
            rel_idx = relation2index[rel]
            obj_idx = constant2index[obj]

            try:
                pair_idx = pair2index[(sub_idx, obj_idx)]        
            except:
                pair_idx = pair2index[(obj_idx, sub_idx)]
                rel_idx = rel_idx + len(relations)
            if label == "1":
                labels_train[pair_idx][len(types) + rel_idx] = 1
            elif label == "0":
                labels_train[pair_idx][len(types) + rel_idx] = 0
            masks_train[pair_idx][len(types) + rel_idx] = 1 

        for triple in valid_type_triples_with_label:
            cons = triple[0]
            typ = triple[2]
            label = triple[3]
            pair_idx= pair2index[(constant2index[cons], constant2index[cons])]
            typ_idx = type2index[typ]
            if label == "1":
                labels_valid[pair_idx][typ_idx] = 1
            elif label == "0":
                labels_valid[pair_idx][typ_idx] = 0
            masks_valid[pair_idx][typ_idx] = 1

        for triple in valid_real_triples_with_label:
            sub = triple[0]
            rel = triple[1]
            obj = triple[2]
            label = triple[3]
            sub_idx = constant2index[sub]
            rel_idx = relation2index[rel]
            obj_idx = constant2index[obj]

            try:
                pair_idx = pair2index[(sub_idx, obj_idx)]        
            except:
                pair_idx = pair2index[(obj_idx, sub_idx)]
                rel_idx = rel_idx + len(relations)
            if label == "1":
                labels_valid[pair_idx][len(types) + rel_idx] = 1
            elif label == "0":
                labels_valid[pair_idx][len(types) + rel_idx] = 0
            masks_valid[pair_idx][len(types) + rel_idx] = 1 
        
        #features are only generated for known train/valid set (pos&neg)
        
        for triple in all_known_type_triples_with_label:
            cons = triple[0]
            typ = triple[2]
            label = triple[3]
            pair_idx= pair2index[(constant2index[cons], constant2index[cons])]
            typ_idx = type2index[typ]
            if label == "1":
                features[pair_idx][typ_idx] = 1
            elif label == "0":
                features[pair_idx][typ_idx] = 0


        for triple in all_known_real_triples_with_label:
            sub = triple[0]
            rel = triple[1]
            obj = triple[2]
            label = triple[3]
            sub_idx = constant2index[sub]
            rel_idx = relation2index[rel]
            obj_idx = constant2index[obj]

            try:
                pair_idx = pair2index[(sub_idx, obj_idx)]        
            except:
                pair_idx = pair2index[(obj_idx, sub_idx)]
                rel_idx = rel_idx + len(relations)
            if label == "1":
                features[pair_idx][len(types) + rel_idx] = 1
            elif label == "0":
                features[pair_idx][len(types) + rel_idx] = 0

        features.requires_grad = True
        labels_train.requires_grad = False
        labels_valid.requires_grad = False

        return features, labels_train, labels_valid, masks_train, masks_valid
    
    features, labels_train, labels_valid, masks_train, masks_valid = initialize(train_real_triples_with_label, train_type_triples_with_label, valid_real_triples_with_label, valid_type_triples_with_label)
    
    num_type = len(types)
    num_relation = len(relations)
    
    print("Finished generation")
    
    print("Total time elapsed for encoding: {:.4f}s".format(time.time() - t_start))
    
    return adj, features, labels_train, labels_valid, masks_train, masks_valid, num_type, num_relation