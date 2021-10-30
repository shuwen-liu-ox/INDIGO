import os
import argparse
import random
from utils.tool import dictionary

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', '-d', type=str, required=True,
                    help='Name of dataset')
parser.add_argument('--sample_rate', '-s', type=int, default=3,
                    help='Number of negative examples generated for one positive example')

args = parser.parse_args()

f1 = open("data/{}/train/train.txt".format(args.dataset))
f2 = open("data/{}/train/valid.txt".format(args.dataset))

triples_train_normal = []
triples_train_type = []
triples_valid_normal = []
triples_valid_type = []

triples_train = []
triples_valid = []

triples = []

entity_set = set()
relation_set = set()
type_set = set()
pair_set = set()

for line in f1:
    triple = line.strip().split("\t")
    triples.append(triple)
    triples_train.append(triple)
    if triple[1] == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
        triples_train_type.append(triple)
        entity_set.add(triple[0])
        type_set.add(triple[2])
    else:
        triples_train_normal.append(triple)
        entity_set.add(triple[0])
        entity_set.add(triple[2])
        relation_set.add(triple[1])
        pair_set.add((triple[0],triple[2]))

for line in f2:
    triple = line.strip().split("\t")
    triples.append(triple)
    triples_valid.append(triple)
    if triple[1] == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
        triples_valid_type.append(triple)
        entity_set.add(triple[0])
        type_set.add(triple[2])
    else:
        triples_valid_normal.append(triple)
        entity_set.add(triple[0])
        entity_set.add(triple[2])
        relation_set.add(triple[1])
        pair_set.add((triple[0],triple[2]))
        

entities = list(entity_set)
entity2index = dictionary(entities)
relations = list(relation_set)
relation2index = dictionary(relations)
pairs = list(pair_set)
pair2index = dictionary(pairs)
types = list(type_set)
type2index = dictionary(types)

joint_relations = dict([(i,set()) for i in range(len(relations))])
joint_types = dict([(i,set()) for i in range(len(types))])

function_relation_dic = []
for j in range(len(relations)):
    function_relation_dic.append(dict([(i, 0) for i in range(len(entities))]))
reverse_function_relation_dic = []
for j in range(len(relations)):
    reverse_function_relation_dic.append(dict([(i, 0) for i in range(len(entities))]))
    
for triple in triples:
    if triple[1] != "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
        sub_id = entity2index[triple[0]]
        obj_id = entity2index[triple[2]]
        rel_id = relation2index[triple[1]]
        function_relation_dic[rel_id][sub_id] += 1
        reverse_function_relation_dic[rel_id][obj_id] += 1

functional_relations = []
for i in range(len(relations)):
    flag = 1
    for j in range(len(entities)):
        if function_relation_dic[i][j] > 1:
            flag = 0
    if flag == 1:
        functional_relations.append(relations[i])
        
reverse_functional_relations = []
for i in range(len(relations)):
    flag = 1
    for j in range(len(entities)):
        if reverse_function_relation_dic[i][j] > 1:
            flag = 0
    if flag == 1:
        reverse_functional_relations.append(relations[i])
        
joint_relations = dict([(i,set()) for i in range(len(relations))])
joint_types = dict([(i,set()) for i in range(len(types))])
types_for_entity = dict([(i,set()) for i in range(len(entities))])
rels_for_pair = dict([(i,set()) for i in range(len(pairs))])


for triple in triples:
    if triple[1] != "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
        pair_id = pair2index[(triple[0],triple[2])]
        rel_id = relation2index[triple[1]]
        rels_for_pair[pair_id].add(rel_id)
    else:
        entity_id = entity2index[triple[0]]
        type_id = type2index[triple[2]]
        types_for_entity[entity_id].add(type_id)
    

for i in range(len(pairs)):
    colist = rels_for_pair[i]
    for r1 in colist:
        for r2 in colist:
            if r1 != r2:
                joint_relations[r1].add(r2)
                joint_relations[r2].add(r1)
for i in range(len(entities)):
    colist = types_for_entity[i]
    for type1 in colist:
        for type2 in colist:
            if type1 != type2:
                joint_types[type1].add(type2)
                joint_types[type2].add(type1)
                
                
import random
train_path = "data/{}/train/train-labeled.txt".format(args.dataset)
valid_path = "data/{}/train/valid-labeled.txt".format(args.dataset)

def generate(path, g_triples):
    count = 0
    f3 = open(path, "w+")
    for triple in g_triples:
        h = triple[0]
        r = triple[1]
        t = triple[2]
        f3.write("{}\t{}\t{}\t1\n".format(h, r, t))
        if r in functional_relations:
            count += 1
            neg_ts = random.sample(entities, 50)
            neg_count = 0
            for neg_t in neg_ts:
                if neg_t!= t:
                    neg_count += 1
                    negative = [h, r, neg_t]

                    f3.write("{}\t{}\t{}\t0\n".format(negative[0],negative[1], negative[2]))
                    if neg_count == args.sample_rate:
                        break
        if r in reverse_functional_relations:
            count += 1
            neg_hs = random.sample(entities, 50)
            neg_count = 0
            for neg_h in neg_hs:
                if neg_h != h:
                    neg_count += 1
                    negative = [neg_h, r, t]
                    f3.write("{}\t{}\t{}\t0\n".format(negative[0],negative[1], negative[2]))
                    if neg_count == args.sample_rate:
                        break
        #neg_r = random.sample(joint_relations[relation2index[r]], 1)
        if r != "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
            rel_id = relation2index[r]
            candidates = [i for i in range(len(relations)) if (i not in joint_relations[rel_id]) and (i != rel_id)]
            neg_rs_id = random.sample(candidates, args.sample_rate)
            for neg_r_id in neg_rs_id:
                neg_r = relations[neg_r_id]
                if neg_r != r:
                    negative = [h, neg_r, t]
                    f3.write("{}\t{}\t{}\t0\n".format(negative[0],negative[1], negative[2]))
                else:
                    print("wrong")
        else:
            
            type_id = type2index[t]
            candidates = [i for i in range(len(types)) if (i not in joint_types[type_id]) and (i != type_id)]
            try:
                candidates = random.sample(candidates, 3)
                for Type in candidates:
                    f3.write("{}\t{}\t{}\t0\n".format(h, "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>", types[Type]))
            except:
                candidates = ["<http://rdf.freebase.com/ns/non-sense-type1>", "<http://rdf.freebase.com/ns/non-sense-type2>", "<http://rdf.freebase.com/ns/non-sense-type3>"]
                for Type in candidates:
                    f3.write("{}\t{}\t{}\t0\n".format(h, "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>", Type))
    f3.close()

generate(train_path, triples_train)
generate(valid_path, triples_valid)