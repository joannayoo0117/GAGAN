import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GateAugmentedGraphAttentionLayer as GAGAL
from layers import DistanceAwareAdjacencyMatrix as DAAM
from layers import GraphAttentionLayer as GAL


class GAGAN(nn.Module):
    def __init__(self, n_out, n_feat, n_attns, n_dense, 
                 dim_attn, dim_dense, dropout):
        super(GAGAN, self).__init__()

        self.adjacency_matrix_transformer = DAAM()
        self.dropout = dropout

        self.attentions = [GAGAL(n_feat=n_feat, n_hid=dim_attn, dropout=dropout)]
        self.attentions += \
            [GAGAL(n_feat=dim_attn, n_hid=dim_attn, dropout=dropout) 
            for _ in range(n_attns-1)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.dense = [nn.Linear(dim_attn, dim_dense)]
        self.dense += [
            nn.Linear(dim_dense, dim_dense)
            for i in range(n_dense-1)]

        for i, layer in enumerate(self.dense):
            self.add_module('dense_{}'.format(i), layer)

        self.out_layer = nn.Linear(dim_dense, n_out)


    def forward(self, X, A, A2):
        # TODO Turning off adjacency matrix transformer
        #A2 = self.adjacency_matrix_transformer(A, D)

        X2 = X.clone().detach()

        X = F.dropout(X, self.dropout, training=self.training)
        X2 = F.dropout(X2, self.dropout, training=self.training)

        # TODO Try multihead attn instead of linear?
        for attn in self.attentions:
            X = attn(X=X, A=A)
            X2 = attn(X=X2, A=A2)

        X_graph = torch.mean(X2-X, dim=0).unsqueeze(0)

        for layer in self.dense:
            X_graph = F.dropout(X_graph, self.dropout, training=self.training)
            X_graph = layer(X_graph)
            X_graph = F.relu(X_graph)

        out =  self.out_layer(X_graph)

        return F.softmax(out, dim=1)


class GAT(nn.Module):
    """
    Acknowledgement: https://github.com/Diego999/pyGAT
    """
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,
                nlinear, nhid_linear, use_distance_aware_adj):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [
            GAL(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) 
            for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GAL(nhid * nheads, nhid_linear, 
            dropout=dropout, alpha=alpha, concat=False)

        self.dense = [
            nn.Linear(nhid_linear, nhid_linear)
            for i in range(nlinear)]

        for i, layer in enumerate(self.dense):
            self.add_module('dense_{}'.format(i), layer)

        self.out_layer = nn.Linear(nhid_linear, nclass)
        self.use_distance_aware_adj = use_distance_aware_adj


    def forward(self, X, A, A2):
        adj = None
        if self.use_distance_aware_adj: adj = A2
        else: adj = A

        X = F.dropout(X, self.dropout, training=self.training)
        X = torch.cat([att(X, adj) for att in self.attentions], dim=1)
        X = F.dropout(X, self.dropout, training=self.training)
        X = self.out_att(X, adj)
        X_graph = torch.mean(X, dim=0).unsqueeze(0)

        for layer in self.dense:
            X_graph = layer(X_graph)
            X_graph = F.relu(X_graph)
            X_graph = F.dropout(X_graph, self.dropout, training=self.training)

        out =  self.out_layer(X_graph)

        return F.softmax(out, dim=1)

"""
class GCN(nn.Module):
    # pytorch implementation of deepchem.models.graph_models.GraphConvModel

    def __init__(self, nconvs, dim_conv, ndense, dim_dense, dropout, nfeat):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.graph_convs = [
            GraphConv(nfeat, dim_conv, dropout=dropout) for _ in range(convs)]
        self.graph_pool = GraphPool()
        # self.dense = nn.Linear() # what is the size??
        self.neural_fingerprint = GraphGather()

    def forward(self, features):
        X, deg_slice, membership, deg_adj_list = features
        for gc in self.graph_convs:
            X = gc()
"""