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


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W) #500 x 140
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class DistanceAwareAdjacencyMatrix(nn.Module):
    """
    Note: 
    This is not currently being used.
    """
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