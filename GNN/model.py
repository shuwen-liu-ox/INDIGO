import torch
import torch.nn as nn
import torch.nn.functional as F
from GNN.layer import GraphConvolutionLayer

class GCN(nn.Module):
    
    """
    a GCN model with multiple layers
    """
    
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
            
        #in our case, nfeat = nclass
        
        self.gc1 = GraphConvolutionLayer(nfeat, nhid)
        self.gc2 = GraphConvolutionLayer(nhid, nclass)
        #self.gc2 = GraphConvolutionLayer(nhid, nhid2)
        #self.gc3 = GraphConvolutionLayer(nhid2, nclass)
        
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        
        """
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        """
        #output with applying a sigmoid function
        x = torch.sigmoid(x)
        output = torch.reshape(x,(adj.shape[0], -1))
        
        return output