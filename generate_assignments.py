import numpy as np
import os
import argparse

#parse the argument

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, required=True,
                    help='Name of dataset')
args = parser.parse_args()

# R(x,y) --> S(x,y)
rule_path = 'data/rule/{}'.format(args.dataset)
pattern1_rels = []
ts = []
for line in open('{}/pattern1.txt'.format(rule_path)):
    t = line.strip().split('\t')
    ts.append(t)
count = 0
count1 = 0
count9_1 = 0
count8_9 = 0
count7_8 = 0
i = 0
for t in ts:
    confidence = float(t[0])
    r = t[1]
    s = t[2]
    p = 'pattern1'
    if confidence == 1:
        save_path = "{}/{}/confidence1".format(rule_path, p)
        count1 += 1
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif 0.9 <= confidence < 1 :
        count9_1 += 1
        save_path = "{}/{}/confidence0.9-1".format(rule_path, p)
        if not os.path.exists(save_path):
            os.makedirs(save_path)   
    elif 0.8 <= confidence < 0.9 :
        count8_9 += 1
        save_path = "{}/{}/confidence0.8-0.9".format(rule_path, p)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif 0.7 <= confidence < 0.8 :
        count7_8 += 1
        save_path = "{}/{}/confidence0.7-0.8".format(rule_path, p)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        continue
    f1 = open('{}/test-graph-{}.txt'.format(save_path, str(i)),'w+')
    for x in ['a']:
        for y in ['a', 'b']:
            f1.write('{}\t{}\t{}\n'.format(x, r, y))
    f2 = open('{}/test-fact-{}.txt'.format(save_path, str(i)),'w+')
    for x in ['a']:
        for y in ['a', 'b']:
            f2.write('{}\t{}\t{}\t1\n'.format(x, s, y))
    f1.close()
    f2.close()
    i += 1



# R(x,y) -> R(y,x)
pattern3_rels = []
ts = []
for line in open('{}/pattern3.txt'.format(rule_path)):
    t = line.strip().split('\t')
    ts.append(t)
count = 0
count1 = 0
count9_1 = 0
count8_9 = 0
count7_8 = 0

i = 0
for t in ts:
    confidence = float(t[0])
    r = t[1]
    p = 'pattern3'
    if confidence == 1:
        save_path = "{}/{}/confidence1".format(rule_path, p)
        count1 += 1
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif 0.9 <= confidence < 1 :
        count9_1 += 1
        save_path = "{}/{}/confidence0.9-1".format(rule_path, p)
        if not os.path.exists(save_path):
            os.makedirs(save_path)   
    elif 0.8 <= confidence < 0.9 :
        count8_9 += 1
        save_path = "{}/{}/confidence0.8-0.9".format(rule_path, p)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif 0.7 <= confidence < 0.8 :
        count7_8 += 1
        save_path = "{}/{}/confidence0.7-0.8".format(rule_path, p)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        continue
    f1 = open('{}/test-graph-{}.txt'.format(save_path, str(i)),'w+')
    for x in ['a']:
        for y in ['a', 'b']:
            f1.write('{}\t{}\t{}\n'.format(x, r, y))
    f2 = open('{}/test-fact-{}.txt'.format(save_path, str(i)),'w+')
    for x in ['a']:
        for y in ['a', 'b']:
            f2.write('{}\t{}\t{}\t1\n'.format(y, r, x))
    f1.close()
    f2.close()
    i += 1


# R(x,y), S(y,z) -> T(x,z) 

pattern2_rels = []
ts = []
for line in open('{}/pattern4.txt'.format(rule_path)):
    t = line.strip().split('\t')
    ts.append(t)
count = 0
count1 = 0
count9_1 = 0
count8_9 = 0
count7_8 = 0
count6_7 = 0
count5_6 = 0
i = 0
for t in ts:
    confidence = float(t[0])
    r = t[1]
    s = t[2]
    t = t[3]
    p = 'pattern4'
    if confidence == 1:
        save_path = "{}/{}/confidence1".format(rule_path, p)
        count1 += 1
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif 0.9 <= confidence < 1 :
        count9_1 += 1
        save_path = "{}/{}/confidence0.9-1".format(rule_path, p)
        if not os.path.exists(save_path):
            os.makedirs(save_path)   
    elif 0.8 <= confidence < 0.9 :
        count8_9 += 1
        save_path = "{}/{}/confidence0.8-0.9".format(rule_path, p)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif 0.7 <= confidence < 0.8 :
        count7_8 += 1
        save_path = "{}/{}/confidence0.7-0.8".format(rule_path, p)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        continue
    
    f1 = open('{}/test-graph-{}.txt'.format(save_path, str(i)),'w+')
    f2 = open('{}/test-fact-{}.txt'.format(save_path, str(i)),'w+')
    for x in ['a']:
        for y in ['b']:
            for z in ['a', 'c']:
                
                f1.write('{}\t{}\t{}\n'.format(x, r, y))
                f1.write('{}\t{}\t{}\n'.format(y, s, z))


                f2.write('{}\t{}\t{}\t1\n'.format(x, t, z))
    f1.close()
    f2.close()
    i += 1



#R(x,y) , S(x,y) -> T(x,y)


ts = []
for line in open('{}/pattern5.txt'.format(rule_path)):
    t = line.strip().split('\t')
    ts.append(t)
count = 0
count1 = 0
count9_1 = 0
count8_9 = 0
count7_8 = 0
count6_7 = 0
count5_6 = 0
i = 0
for t in ts:
    confidence = float(t[0])
    r = t[1]
    s = t[2]
    t = t[3]
    p = 'pattern5'
    if confidence == 1:
        save_path = "{}/{}/confidence1".format(rule_path, p)
        count1 += 1
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif 0.9 <= confidence < 1 :
        count9_1 += 1
        save_path = "{}/{}/confidence0.9-1".format(rule_path, p)
        if not os.path.exists(save_path):
            os.makedirs(save_path)   
    elif 0.8 <= confidence < 0.9 :
        count8_9 += 1
        save_path = "{}/{}/confidence0.8-0.9".format(rule_path, p)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif 0.7 <= confidence < 0.8 :
        count7_8 += 1
        save_path = "{}/{}/confidence0.7-0.8".format(rule_path, p)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        continue
    
    
    f1 = open('{}/test-graph-{}.txt'.format(save_path, str(i)),'w+')
    for x in ['a']:
        for y in ['a', 'b']:
            f1.write('{}\t{}\t{}\n'.format(x, r, y))
            f1.write('{}\t{}\t{}\n'.format(x, s, y))
    f2 = open('{}/test-fact-{}.txt'.format(save_path, str(i)),'w+')
    for x in ['a']:
        for y in ['a', 'b']:
            f2.write('{}\t{}\t{}\t1\n'.format(x, t, y))
    f1.close()
    f2.close()
    i += 1


# A(x) -> B(x)


ts = []
for line in open('{}/pattern2.txt'.format(rule_path)):
    t = line.strip().split('\t')
    ts.append(t)
count = 0
count1 = 0
count9_1 = 0
count8_9 = 0
count7_8 = 0
count6_7 = 0
count5_6 = 0
i = 0
for t in ts:
    confidence = float(t[0])
    a = t[1]
    b = t[2]
    p = 'pattern2'
    if confidence == 1:
        save_path = "{}/{}/confidence1".format(rule_path, p)
        count1 += 1
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif 0.9 <= confidence < 1 :
        count9_1 += 1
        save_path = "{}/{}/confidence0.9-1".format(rule_path, p)
        if not os.path.exists(save_path):
            os.makedirs(save_path)   
    elif 0.8 <= confidence < 0.9 :
        count8_9 += 1
        save_path = "{}/{}/confidence0.8-0.9".format(rule_path, p)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif 0.7 <= confidence < 0.8 :
        count7_8 += 1
        save_path = "{}/{}/confidence0.7-0.8".format(rule_path, p)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        continue
    
    
    f1 = open('{}/test-graph-{}.txt'.format(save_path, str(i)),'w+')
    for x in ['a']:
        f1.write('{}\t{}\t{}\n'.format(x, '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>', a))
        f1.write('c\t/sports/sports_team/roster./american_football/football_historical_roster_position/position_s\td\n')
    f2 = open('{}/test-fact-{}.txt'.format(save_path, str(i)),'w+')
    for x in ['a']:
        f2.write('{}\t{}\t{}\t1\n'.format(x, '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>', b))
    f1.close()
    f2.close()
    i += 1


# A1(x), A2(x) -> B(x)


ts = []
for line in open('{}/pattern6.txt'.format(rule_path)):
    t = line.strip().split('\t')
    ts.append(t)
count = 0
count1 = 0
count9_1 = 0
count8_9 = 0
count7_8 = 0
count6_7 = 0
count5_6 = 0
i = 0
for t in ts:
    confidence = float(t[0])
    a1 = t[1]
    a2 = t[2]
    b = t[3]
    p = 'pattern6'
    if confidence == 1:
        save_path = "{}/{}/confidence1".format(rule_path, p)
        count1 += 1
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif 0.9 <= confidence < 1 :
        count9_1 += 1
        save_path = "{}/{}/confidence0.9-1".format(rule_path, p)
        if not os.path.exists(save_path):
            os.makedirs(save_path)   
    elif 0.8 <= confidence < 0.9 :
        count8_9 += 1
        save_path = "{}/{}/confidence0.8-0.9".format(rule_path, p)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif 0.7 <= confidence < 0.8 :
        count7_8 += 1
        save_path = "{}/{}/confidence0.7-0.8".format(rule_path, p)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        continue
    
    
    f1 = open('{}/test-graph-{}.txt'.format(save_path, str(i)),'w+')
    for x in ['a']:
        f1.write('{}\t{}\t{}\n'.format(x, '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>', a1))
        f1.write('{}\t{}\t{}\n'.format(x, '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>', a2))
        f1.write('c\t/sports/sports_team/roster./american_football/football_historical_roster_position/position_s\td\n')
    f2 = open('{}/test-fact-{}.txt'.format(save_path, str(i)),'w+')
    for x in ['a']:
        f2.write('{}\t{}\t{}\t1\n'.format(x, '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>', b))
    f1.close()
    f2.close()
    i += 1
