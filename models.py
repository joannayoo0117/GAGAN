import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GateAugmentedGraphAttentionLayer as GAGAL
from layers import DistanceAwareAdjacencyMatrix as DAAM



class Model(nn.Module):
    def __init__(self, n_out, n_feat, n_attns, n_dense, 
                 dim_attn, dim_dense, dropout):
        super(Model, self).__init__()

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

        X_graph = torch.sum(X2-X, dim=0)

        for layer in self.dense:
            X_graph = F.dropout(X_graph, self.dropout, training=self.training)
            X_graph = layer(X_graph)
            X_graph = F.relu(X_graph)

        out =  self.out_layer(X_graph)

        return F.sigmoid(out)