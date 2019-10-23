import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GateAugmentedGraphAttentionLayer(nn.Module):
    """
    https://github.com/Diego999/pyGAT/blob/master/models.py
    https://arxiv.org/abs/1904.08144
    """

    def __init__(self, n_feat, n_hid, dropout=0.3):
        super(GateAugmentedGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.n_feat = n_feat
        self.n_hid = n_hid

        self.W = nn.Parameter(torch.zeros(size=(n_feat, n_hid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.E = nn.Parameter(torch.zeros(size=(n_hid, n_hid)))
        nn.init.xavier_uniform_(self.E.data, gain=1.414)
        self.U = nn.Parameter(torch.zeros(size=(n_feat + n_hid, 1)))
        nn.init.xavier_uniform_(self.U.data, gain=1.414)
        self.b = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)


    def forward(self, X, A):
        N = X.size()[0]
        h = torch.mm(X, self.W)

        # symmetric attention as in paper
        attention = torch.mm(torch.mm(h, self.E), 
                             torch.transpose(h, 0, 1)) + \
                    torch.mm(torch.mm(h, torch.transpose(self.E, 0, 1)), 
                             torch.transpose(h, 0, 1))
        # softmax just on the nodes connected together
        mask = (A > 0).float()
        attention = torch.mul(F.softmax(torch.mul(attention, mask)), A)
        attention = F.dropout(attention, self.dropout)
        h_prime = torch.matmul(attention, h)

        # gate augmentation
        gate = F.sigmoid(torch.mm(torch.cat([X.squeeze(), h], dim=1), self.U)\
            + torch.cat(N * [self.b]).view((N, 1)))
        X_out = torch.mul(gate, h) \
            + torch.mul((torch.ones(gate.size()).type_as(gate) - gate), h_prime)
        
        return X_out


class DistanceAwareAdjacencyMatrix(nn.Module):
    def __init__(self, max_dist=5):
        super(DistanceAwareAdjacencyMatrix, self).__init__()

        self.mu = torch.nn.Parameter(torch.zeros(size=(1,)))
        self.sigma = torch.nn.Parameter(torch.ones(size=(1,)))

    def forward(self, A, D):
        """
        Input:
            A: Adjacency Matrix.
            torch.Tensor of shape (dim_c, dim_c)
                where dim_c = dim_l + dim_p
            D: Distance Matrix.
            torch.Tensor of shape (dim_l, dim_p)
                where D[i, j] represents the distance of ligand atom i and 
                protein atom j in graph
                D[i, j] = 0 if i and j not connected
        Output:
            A2: Adjacency Matrix.
        """
        mask = (D > 0).float()
        D = torch.exp(-(D - self.mu.squeeze()) ** 2)\
            / self.sigma.squeeze()
        D = (D * mask)

        (dim_l, dim_p) = D.size()

        A2 = A.clone().detach()
        A2[-dim_l:, :dim_p].data = D.data
        A2[:dim_p, -dim_l:] = torch.transpose(D, 0, 1)

        return A2