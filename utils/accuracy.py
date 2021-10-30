import torch
import numpy as np
from sklearn import metrics

def compute_rank_metrics(output_now, hits_true_now, hits_candidates_now):
    
    """
    Compute ranking-based metrics
    """
    ranks = []

    t_idx = 0
    for (pair_idx, dim_idx) in hits_true_now:
        scores = []
        scores.append(output_now[pair_idx][dim_idx].item())
        
        for c in hits_candidates_now[t_idx]:
            pair_false_idx = c[0]
            dim_false_idx = c[1]
            scores.append(output_now[pair_false_idx][dim_false_idx].item())    

        rank = len(scores) - np.argwhere(np.argsort(scores) == 0)
        ranks.append(rank)  
        t_idx += 1
    

    mr = np.mean(ranks)
    mrr = np.mean(1 / np.array(ranks))
    
    isHit1List = [x for x in ranks if x <= 1]
    isHit3List = [x for x in ranks if x <= 3]
    isHit10List = [x for x in ranks if x <= 10]
    hits_1 = len(isHit1List) / len(ranks)
    hits_3 = float(len(isHit3List)) / float(len(ranks))
    hits_10 = len(isHit10List) / len(ranks)
    return mr, mrr, hits_1, hits_3, hits_10

def compute_accuracy_for_train(output, labels, mask, score_threshold, num_type, num_relation):
    """
    Compute:
    precision, recall, f1-score
    """

    #compute precision, recall and f1-score
    
    preds = torch.gt(output, score_threshold)    
    preds = torch.mul(preds, mask)

    preds_sum = 0
    labels_sum = 0
    correct_sum = 0
    case_sum = 0

    correct = torch.mul(preds.eq(labels), mask)
    mask_positive = torch.eq(preds, 1)
    positive_correct = torch.mul(correct , mask_positive)
    positive_correct_num = torch.sum(positive_correct, dim = 1).sum()
    positive_real_num = labels.sum()
    positive_preds_num = preds.sum()
    
    mask_negative = torch.eq(preds, 0)
    mask_negative = torch.mul(mask_negative, mask)
    
    negative_correct = torch.mul(preds.eq(labels), mask_negative)
    negative_correct_num = negative_correct.sum()
    negative_real_num = torch.mul(torch.eq(labels, 0), mask).sum()
    negative_preds_num = torch.mul(torch.eq(preds, 0), mask).sum()
                                            

    true_positive = positive_correct_num
    true_negative = negative_correct_num
    false_positive = positive_preds_num - true_positive
    false_negative = negative_preds_num - true_negative
    
    acc =  (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    precision = (true_positive) / (true_positive + false_positive)
    recall = (true_positive) / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)
    #false_positive_rate = (false_positive) / (false_positive + true_negative)
    #false_negative_rate = (false_negative) / (false_negative + true_positive)
    
    return acc, precision, recall, f1

def compute_accuracy_for_test(output, labels, mask, score_threshold, num_type, num_relation, hits_true, r_hits_candidates):

    """
    Compute:
    precision, recall, f1-score, and AUC
    hits@1, hits@3, hits@10, mrr
    """
    
    #compute precision, recall and f1-score
    
    preds = torch.gt(output, score_threshold)    
    preds = torch.mul(preds, mask)

    preds_sum = 0
    labels_sum = 0
    correct_sum = 0
    case_sum = 0

    correct = torch.mul(preds.eq(labels), mask)
    mask_positive = torch.eq(preds, 1)
    positive_correct = torch.mul(correct , mask_positive)
    positive_correct_num = torch.sum(positive_correct, dim = 1).sum()
    positive_real_num = labels.sum()
    positive_preds_num = preds.sum()
    
    mask_negative = torch.eq(preds, 0)
    mask_negative = torch.mul(mask_negative, mask)
    
    negative_correct = torch.mul(preds.eq(labels), mask_negative)
    negative_correct_num = negative_correct.sum()
    negative_real_num = torch.mul(torch.eq(labels, 0), mask).sum()
    negative_preds_num = torch.mul(torch.eq(preds, 0), mask).sum()
                                            


    true_positive = positive_correct_num
    true_negative = negative_correct_num
    false_positive = positive_preds_num - true_positive
    false_negative = negative_preds_num - true_negative
    
    acc =  (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    precision = (true_positive) / (true_positive + false_positive)
    recall = (true_positive) / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)
    false_positive_rate = (false_positive) / (false_positive + true_negative)
    false_negative_rate = (false_negative) / (false_negative + true_positive)
    

    r_mr, r_mrr, r_hits1, r_hits3, r_hits10 = compute_rank_metrics(output, hits_true, r_hits_candidates)

      
        
    
    #compute AUC
    
    y_pred = []
    y_true = []
    for i,j in mask.nonzero():
        out = output[i][j]
        true = labels[i][j]
        y_pred.append(out)
        y_true.append(true)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    roc_auc = metrics.roc_auc_score(y_true, y_pred)
    auc_pr = metrics.average_precision_score(y_true, y_pred)
    
    return acc, precision, recall, f1, false_positive_rate, false_negative_rate, roc_auc, auc_pr, r_mr, r_mrr, r_hits1, r_hits3, r_hits10