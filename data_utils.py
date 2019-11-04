import torch
from torch.utils.data import Dataset
from utils import get_pdb_features, get_convmol_features, one_hot_encoding
import pandas as pd
import numpy as np
import os

class ConvMolDataset(Dataset):
    def __init__(self, 
                 index_csv='./data/pdbbind_binding_affinity.csv', 
                 data_dir='./data/pdbbind/v2018',
                 num_positive=1000,
                 num_negative=1000,
                 seed=117,
                 binding_type='ic50', 
                 train_test_split=0.05,
                 transform=None):
        data = pd.read_csv(index_csv)
        data = data[data['binding_type'] == binding_type]

        data_positive = data.nlargest(num_positive, 'binding_affinity')
        data_positive['label'] = 1
        data_negative = data.nsmallest(num_negative, 'binding_affinity')
        data_negative['label'] = 0

        self.data_dir = data_dir
        self.data = pd.concat([data_positive, data_negative], axis=0)
        self.data.index = range(self.data.shape[0])
        self.train_test_split = train_test_split
        self.seed = seed

    def __getitem__(self, index):
        pdbid = self.data.loc[index, 'id']
        try:
            (node_feat, deg_slice, membership, deg_adj_list) = get_convmol_features(
                protein_pdb_file="%s/%s_pocket.pdb" % (pdbid, pdbid),
                ligand_pdb_file="%s/%s_ligand.pdb" % (pdbid, pdbid),
                data_dir=self.data_dir)

            label = self.data.loc[index, 'label']
            return (torch.Tensor(node_feat), 
                    torch.Tensor(deg_slice), 
                    torch.Tensor(membership),
                    torch.Tensor(deg_adj_list)), \
                    torch.Tensor([label]).view(-1).long()

        except BaseException as e:
            print('Could not get features for PDBID {}'.format(pdbid))
            print(e)
            pass

    def __len__(self):
        return self.data.shape[0]

    def __nlabels__(self):
        return len(self.data.label.unique().tolist())

    def __nfeats__(self):
        return self.__getitem__(0)[0][0].shape[1]

    def __train_test_split__(self):
        np.random.seed(self.seed)
        indices = self.data.index.tolist()
        dataset_size = len(self)
        split = int(np.floor(self.train_test_split * dataset_size))
        np.random.shuffle(indices)
        train, test, val \
            = indices[2*split:], indices[:split], indices[split:2*split]

        return train, test, val


class PDBBindDataset(Dataset):
    def __init__(self, 
                 index_csv='./data/pdbbind_binding_affinity.csv', 
                 data_dir='./data/pdbbind/v2018',
                 num_positive=1000,
                 num_negative=1000,
                 seed=117,
                 binding_type='ic50', 
                 train_test_split=0.05,
                 transform=None):
        data = pd.read_csv(index_csv)
        data = data[data['binding_type'] == binding_type]

        data_positive = data.nlargest(num_positive, 'binding_affinity')
        data_positive['label'] = 1
        data_negative = data.nsmallest(num_negative, 'binding_affinity')
        data_negative['label'] = 0

        self.data_dir = data_dir
        self.data = pd.concat([data_positive, data_negative], axis=0)
        self.data.index = range(self.data.shape[0])
        self.train_test_split = train_test_split
        self.seed = seed

    def __getitem__(self, index):
        pdbid = self.data.loc[index, 'id']
        try:
            X, A, A2 = get_pdb_features(
                protein_pdb_file="%s/%s_pocket.pdb" % (pdbid, pdbid),
                ligand_pdb_file="%s/%s_ligand.pdb" % (pdbid, pdbid),
                data_dir=self.data_dir)
            # label = one_hot_encoding(self.data.loc[index, 'label'], {0, 1})
            label = self.data.loc[index, 'label']
            return (torch.Tensor(X), torch.Tensor(A), torch.Tensor(A2)), \
                    torch.Tensor([label]).view(-1).long()

        except BaseException as e:
            print('Could not get features for PDBID {}'.format(pdbid))
            print(e)
            pass

    def __len__(self):
        return self.data.shape[0]

    def __nlabels__(self):
        return len(self.data.label.unique().tolist())

    def __nfeats__(self):
        return self.__getitem__(0)[0][0].shape[1]

    def __train_test_split__(self):
        np.random.seed(self.seed)
        indices = self.data.index.tolist()
        dataset_size = len(self)
        split = int(np.floor(self.train_test_split * dataset_size))
        np.random.shuffle(indices)
        train, test, val \
            = indices[2*split:], indices[:split], indices[split:2*split]

        return train, test, val


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)