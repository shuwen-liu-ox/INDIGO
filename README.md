# INDIGO: GNN-Based Inductive Knowledge Graph Completion Using Pair-Wise Encoding

This repository is the official implementation of the paper INDIGO: GNN-Based Inductive Knowledge Graph Completion Using Pair-Wise Encoding.

## Overview

INDIGO is a system for inductive knowledge graph completion, which can generalise to the predictions for any constants not observed in the training process. It contains a pair-wise encoder, a GNN application, and a decoder. Our results show that INDIGO not only outperforms the baselineson these benchmarks, but can also be trained and applied more efficiently.

This repository contains the benchmarks used in the experiments for the paper, as well as the code to reproduce the results.

##  Benchmarks

### Datasets
 
The benchmark datasets are in folder data, which contain a subfolder for each of our 22 benchmarks (see further details in the paper): 
- 12 benchmarks GraIL-BM_DatasetName_version of Teru et al. [2020](https://github.com/kkteru/grail)
- 9 benchmarks Hamaguchi-BM_DatasetName-version of Hamaguchi et al. [2017](https://github.com/takuo-h/GNN-for-OOKB)
- 1 benchmark INDIGO-BM developed by us

Each benchmark subfolder contains two folders, one for training and one for testing:
- train
- test

Each folder train contains 2 files:

- train.txt				% input KG (denoted as calligraphic T in the paper)
- valid.txt				% validation version of train.txt used to select the optimal number of epochs when training

Each folder test contains 2 files and 1 subfolder:

- test-graph.txt			% incomplete KG (denoted as calligraphic K_test in the paper)
- test-fact.txt			% positive examples to predict (Lambda^+_test)
- test-random-sample

Each folder test-random-sample contains 10 files:
- test{0-9}.txt			% positive and negative examples to predict (Lambda^+_test + Lambda^-_test); note that this *is* a part of a benchmark (see the paper for details)


###  Negative sampling

To generate training negative examples for each benchmark using the negative sampling strategy proposed in the paper, run the following command in this folder (i.e., the root):

```time python negative_sampling.py --dataset BENCHMARK_NAME --sample_rate NUM_NEG```

where BENCHMARK_NAME is INDIGO-BM or GraIL-BM_DatasetName_version or Hamaguchi-BM_DatasetName-version for appropriate values of DatasetName (e.g., WN18RR or head) and version (e.g., v3 or 1000), and where NUM_NEG is the number of negative examples generated for each positive example (we have always used 3). As a result, two files train-labeled.txt and valid-labeled.txt will be added in the train folder of the benchmark. Note that the generating process is stochastic, so the result may be different for different runs. 

-------------------------------

## Experiments

### INDIGO
 
The structure of GCN is implemented based on the code from [pygcn](https://github.com/tkipf/pygcn), the PyTorch implementation of Graph Convolutional Networks (GCNs) for semi-supervised classification.

### System requirements

- Python 3.6.9
- Pytorch 1.4.0
- sklearn
- numpy
- scipy

### Training

To train an INDIGO model for a benchmark, run the following command at this folder:

```time python train.py --dataset BENCHMARK_NAME --epoch NUM_EPOCH```

where BENCHMARK_NAME is the name of a benchmark (e.g., INDIGO-BM) and  NUM_EPOCH is the number of epochs the model will be trained for (in our experiments we always take 3000, see the paper for details)

Then, the trained models are saved in folder models/BENCHMARK_NAME/

In each epoch, the system will print on the screen the loss and classification-based metrics for both training and validation, as well as the epoch number which has the highest validation F1 score (which is chosen as the final model as described above).

### Testing

To test an INDIGO model on a benchmark for classfication-based metrics and ranking-based metrics about relations (e.g. r-Hits@k and r-MRR), run the following command at this folder:

```time python test.py --dataset BENCHMARK_NAME --model_dir BENCHMARK_NAME --model_name MODEL_NAME --print```

To test an INDIGO model on a benchmark for ranking-based metrics about entities (e.g. e-Hits@k and e-MRR), run the following command at this folder:

```time python test_e_hits.py --dataset BENCHMARK_NAME --model_dir BENCHMARK_NAME --model_name MODEL_NAME```

where BENCHMARK_NAME is as above and MODEL_NAME is the name of a trained model, which can be taken from folder models/BENCHMARK_NAME/ (e.g., a model at epoch 1000 with learning rate=0.001, weight decay=5e-08, dimension of hidden layer=64 will have name lr0.001_wd5e-08_hidden64_e1000).

The test scores for the metrics will be printed on the screen, while the predicted triples will be saved to file predictions_BENCHMARK_NAME.txt at this folder.

### Capturing logic rules

First, to identify rules with confidence above a certain threshold, run the following command at this folder:

```time python generate_rules.py --dataset BENCHMARK_NAME --confidence CONFIDENCE```

where CONFIDENCE is the minimum confidence value, and in our experiments we take 0.7.

The rules will be saved to the folder data/rule/BENCHMARK_NAME.

Second, to generate datasets for these rules, run the following command at this folder:
    
```time python generate_assignments.py --dataset BENCHMARK_NAME```

The mini-datasets will be saved to the folder data/rule/BENCHMARK_NAME/PATTERN_NAME. We provided the mini-datasets generated for INDIGO-BM and GraIL-BM_nell_v3 in the folder data/rule/INDIGO-BM and data/rule/GraIL-BM_nell_v3 for convenience.

Finally, to check how many rules have been captured by the model, run the following command at this folder:
    
```time python test_pattern.py --dataset BENCHMARK_NAME --pattern PATTERN_NAME --model_dir BENCHMARK_NAME --model_name MODEL_NAME```

The number of rules captured by the model will be printed on the screen.


## Acknowledgement

Please cite the following paper as the reference if you use the INDIGO-BM dataset or the implementation of INDIGO:
```
@inproceedings{INDIGO21,
  author    = {Shuwen Liu and
               Bernardo Cuenca Grau and
               Ian Horrocks and
               Egor V. Kostylev},
  title     = {INDIGO: GNN-Based Inductive Knowledge Graph Completion Using Pair-Wise Encoding},
  booktitle = {{NeurIPS}},
  year      = {2021}
}
```

