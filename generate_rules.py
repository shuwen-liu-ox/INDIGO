import numpy as np
import os
import argparse

#parse the argument

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, required=True,
                    help='Name of dataset')
parser.add_argument('--confidence', type=float, default=0.7,
                    help='confidence threshold')
args = parser.parse_args()

CONFIDENCE = args.confidence

test_graph = []
test_fact = []
G1 = set()
G2 = set()

dataset = 'data/{}/test/'.format(args.dataset)
save_path = 'data/rule/{}'.format(args.dataset)
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
relations_TG1 = set()
relations_TG2 = set()
relations_G1 = set()
relations_G2 = set()
for line in open('{}/test-graph.txt'.format(dataset)):
    t = line.strip().split('\t')
    if t[1] != '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>':
        test_graph.append(t)
        if (t[0], t[1], t[2]) not in G1:
            G1.add((t[0], t[1], t[2]))
        if (t[0], t[1], t[2]) not in G2:
            G2.add((t[0], t[1], t[2]))
    if t[1] != '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>':
        relations_TG1.add(t[1])
        relations_G1.add(t[1])
        relations_G2.add(t[1])
for line in open('{}/test-fact.txt'.format(dataset)):
    t = line.strip().split('\t')
    if t[1] != '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>':
        test_fact.append(t)
        if (t[0], t[1], t[2]) not in G2:
            G2.add((t[0], t[1], t[2]))
    if t[1] != '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>':
        relations_TG2.add(t[1])
        relations_G2.add(t[1])
relations_TG1 = list(relations_TG1)
relations_TG2 = list(relations_TG2)
relations_G1 = list(relations_G1)
relations_G2 = list(relations_G2)

pairs_for_relation_G1 = dict([(i,set()) for i in relations_G1])
pairs_for_relation_G2 = dict([(i,set()) for i in relations_G2])
for t in G1:
    pairs_for_relation_G1[t[1]].add((t[0],t[2]))
for t in G2:
    pairs_for_relation_G2[t[1]].add((t[0],t[2]))
    
# R(x,y) --> S(x,y)
import numpy as np
f = open('{}/pattern1.txt'.format(save_path),"w+")
for i in range(len(relations_TG2)):
    s = relations_TG2[i]
    for j in range(len(relations_TG1)):
        r = relations_TG1[j]
        if s != r:
            #computing N
            N = len(pairs_for_relation_G1[r])
            M = 0
            for pair in pairs_for_relation_G1[r]:
                if pair in pairs_for_relation_G2[s]:
                    M += 1

            confidence = M / N
            if confidence >= CONFIDENCE:
                f.write('{}\t{}\t{}\n'.format(str(M/N), r, s))
f.close()

# R(x,y) --> R(y,x)

f = open('{}/pattern3.txt'.format(save_path),"w+")
for i in range(len(relations_TG2)):

    s = relations_TG2[i]
    for j in range(len(relations_TG1)):
        r = relations_TG1[j]
        if s == r:
            N = len(pairs_for_relation_G1[r])
            M = 0
            for pair in pairs_for_relation_G1[r]:
                if (pair[1], pair[0]) in pairs_for_relation_G2[s]:
                    M += 1

            confidence = M / N
            if confidence >= CONFIDENCE:

                f.write('{}\t{}\n'.format(str(confidence), r))
f.close()            



# R(x,y), S(y,z) -> T(x,z) 

rs_list = []
for r in relations_TG1:
    for s in relations_TG1:
        rs_list.append((r,s))
def compute_N_for_rs(r, s, G):
    curr_N = 0
    curr_XZ = set()
    for p1 in pairs_for_relation_G1[r]:
        for p2 in pairs_for_relation_G1[s]:
            if p1[1] == p2[0]:
                if (p1[0], p2[1]) not in curr_XZ:
                    curr_N += 1
                    curr_XZ.add((p1[0], p2[1]))
    return curr_N, curr_XZ

import time
time1 = time.time()
N_for_rs = dict([(i,0) for i in rs_list])
XZ_for_rs = dict([(i,set()) for i in rs_list])
i = 0 
for r in relations_TG1:
    i += 1
    for s in relations_TG1:
        N, XZ = compute_N_for_rs(r, s, G1)
        N_for_rs[(r,s)] = N
        XZ_for_rs[(r,s)] = XZ


import numpy as np
f = open('{}/pattern4.txt'.format(save_path),"w+")

for i in range(len(relations_TG2)):
    t = relations_TG2[i]
    for j in range(len(relations_TG1)):
        r = relations_TG1[j]
        for k in range(len(relations_TG1)):
            s = relations_TG1[k]
            #computing N
            N = N_for_rs[(r,s)]
            XZ = XZ_for_rs[(r,s)]
            M = 0
            for pair in pairs_for_relation_G2[t]:
                if pair in XZ:
                    M += 1
            if N == 0:
                continue
            

            confidence = M / N
            if confidence >= CONFIDENCE:
                f.write('{}\t{}\t{}\t{}\n'.format(str(M/N), r,s,t))
f.close()

rs_list = []
for r in relations_TG1:
    for s in relations_TG1:
        rs_list.append((r,s))
def compute_N_for_rs_same(r, s, G):
    curr_N = 0
    curr_XY = set()
    for p1 in pairs_for_relation_G1[r]:
        for p2 in pairs_for_relation_G1[s]:
            if p1[0] == p2[0]:
                if p1[1] == p2[1]:
                    if (p1[0], p1[1]) not in curr_XY:
                        curr_N += 1
                        curr_XY.add((p1[0], p1[1]))
    return curr_N, curr_XY

import time
time1 = time.time()
N_for_rs_same = dict([(i,0) for i in rs_list])
XY_for_rs_same = dict([(i,set()) for i in rs_list])
i = 0 
for r in relations_TG1:
    i += 1
    for s in relations_TG1:
        N, XY = compute_N_for_rs_same(r, s, G1)
        N_for_rs_same[(r,s)] = N
        XY_for_rs_same[(r,s)] = XY

#R(x,y) , S(x,y) -> T(x,y)
f = open('{}/pattern5.txt'.format(save_path),"w+")

for i in range(len(relations_TG2)):

    t = relations_TG2[i]
    for j in range(len(relations_TG1)):
        r = relations_TG1[j]
        for k in range(len(relations_TG1)):
            
                s = relations_TG1[k]
                if s !=r and s != t and r != t:
                    N = N_for_rs_same[(r,s)]
                    XY = XY_for_rs_same[(r,s)]
                    M = 0
                    for pair in pairs_for_relation_G2[t]:
                        if pair in XY:
                            M += 1

                    if N == 0:
                        continue
                    confidence = M / N
                    if confidence >= CONFIDENCE:
                        f.write('{}\t{}\t{}\t{}\n'.format(str(M/N), r,s,t))
f.close()

types_G1 = set()
types_G2 = set()
types_TG1 = set()
types_TG2 = set()
G1_type_triples = set()
G2_type_triples = set()
for line in open('{}/test-graph.txt'.format(dataset)):
    t = line.strip().split('\t')
    if t[1] == '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>':
        types_G1.add(t[2])
        types_G2.add(t[2])
        types_TG1.add(t[2])
        G1_type_triples.add((t[0], t[1], t[2]))
        G2_type_triples.add((t[0], t[1], t[2]))
for line in open('{}/test-fact.txt'.format(dataset)):
    t = line.strip().split('\t')
    if t[1] == '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>':
        types_G2.add(t[2])
        types_TG2.add(t[2])
        G2_type_triples.add((t[0], t[1], t[2]))
        
# A(x) -> B(x)
#Type(a, A) -> Type(a, B)
f = open('{}/pattern2.txt'.format(save_path), "w+")
entities_for_type_G1 = dict([(i,set()) for i in types_G1])
for t in G1_type_triples:
    if t[1] == '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>':
        entities_for_type_G1[t[2]].add(t[0])
entities_for_type_G2 = dict([(i,set()) for i in types_G2])
for t in G2_type_triples:
    if t[1] == '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>':
        entities_for_type_G2[t[2]].add(t[0])   
for b in types_TG2:
    
    for a in types_TG1:
        if a != b:
            M = 0
            N = len(entities_for_type_G1[a])
            for entity in entities_for_type_G2[b]:
                if entity in entities_for_type_G1[a]:
                    M += 1
            if N == 0:
                    continue
            confidence = M / N
            if confidence >= CONFIDENCE:
                f.write('{}\t{}\t{}\n'.format(str(M/N), a,b))
f.close()

# A1(x), A2(x) -> B(x)

f = open('{}/pattern6.txt'.format(save_path),"w+")
entities_for_type_G1 = dict([(i,set()) for i in types_G1])
for t in G1_type_triples:
    if t[1] == '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>':
        entities_for_type_G1[t[2]].add(t[0])
entities_for_type_G2 = dict([(i,set()) for i in types_G2])
for t in G2_type_triples:
    if t[1] == '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>':
        entities_for_type_G2[t[2]].add(t[0])   
for b in types_TG2:
    
    for a1 in types_TG1:
        for a2 in types_TG1:
            if a1 != a2 and a1 != b and a2!=b:
                M = 0
                entities_a1_a2 = entities_for_type_G1[a1] & entities_for_type_G1[a2]
                N = len(entities_a1_a2)
                for entity in entities_for_type_G2[b]:
                    if entity in entities_a1_a2:
                        M += 1
                if N == 0:
                        continue
                confidence = M / N
                if confidence >= CONFIDENCE:
                    f.write('{}\t{}\t{}\t{}\n'.format(str(M/N), a1, a2, b))
f.close()