import torch
from torch.utils.data import Dataset
from utils import get_pdb_features, one_hot_encoding
import pandas as pd
import numpy as np
import os


class DockingDataset(Dataset):
    def __init__(self, 
                 index_csv='./data/dude2pdb.csv',
                 data_dir='./data/docking',
                 num_positive=100,
                 num_negative=100,
                 seed=117,
                 train_test_split=0.05,
                 transform=None):
        self.data_dir = data_dir

        data = self._read_current_docking_files(index_csv, data_dir)
        data = self._split_by_id(data, train_test_split, seed)

        num_pos_test = int(np.floor(train_test_split * num_positive))
        num_neg_test = int(np.floor(train_test_split * num_negative))

        num_pos_train = num_positive - (2*num_pos_test)
        num_neg_train = num_negative - (2*num_neg_test)

        test_df = self._get_n_samples(
            data, 'test', (num_pos_test, num_neg_test))
        val_df = self._get_n_samples(
            data, 'val', (num_pos_test, num_neg_test))
        train_df = self._get_n_samples(
            data, 'train', (num_pos_train, num_neg_train))

        self.data = pd.concat([test_df, val_df, train_df])
        self.data.index = range(self.data.shape[0])


    def __getitem__(self, index):
        id = self.data.loc[index, 'id']
        try:
            X, A, A2 = get_pdb_features(
                protein_pdb_file='%s/%s_pocket.pdb' % (id, id),
                ligand_pdb_file= id + '/' + self.data.loc[index, 'ligands'],
                data_dir=self.data_dir)
            # label = one_hot_encoding(self.data.loc[index, 'label'], {0, 1})
            label = self.data.loc[index, 'label']
            return (torch.Tensor(X), torch.Tensor(A), torch.Tensor(A2)), \
                    torch.Tensor([label]).view(-1).long()

        except BaseException as e:
            print('Could not get features for PDBID {}'.format(id))
            print(e)
            pass

    def __len__(self):
        return self.data.shape[0]

    def __nlabels__(self):
        return len(self.data.label.unique().tolist())

    def __nfeats__(self):
        return self.__getitem__(0)[0][0].shape[1]

    def __train_test_split__(self):
        return (self.data[self.data.subset == 'train'].index.tolist(),
                self.data[self.data.subset == 'val'].index.tolist(),
                self.data[self.data.subset == 'test'].index.tolist(),)

    def _get_n_samples(self, df, type, nums):
        num_pos, num_neg = nums
        sub_df = df[df.subset == type]

        return pd.concat([sub_df[sub_df.label == 1].sample(n=num_pos), 
                          sub_df[sub_df.label == 0].sample(n=num_neg)])
    
    def _split_by_id(self, data, train_test_split, seed):
        np.random.seed(seed)

        ids = np.array(data.id.unique().tolist())
        
        num_test = max(1, int(np.floor(train_test_split * len(ids))))
        num_train = len(ids) - (2 * num_test)
        subset = ['val'] * num_test + ['test'] * num_test \
             + ['train'] * num_train
        np.random.shuffle(subset)
        id2subset = dict(zip(ids, subset))    
        
        data['subset'] = data.id.apply(lambda x: id2subset.get(x, ''))

        return data


    def _read_current_docking_files(self, index_csv, data_dir):
        data = pd.read_csv(index_csv)
        docked_ligands = []
        for pdbid in data.id.tolist():
            if not os.path.isdir(os.path.join(data_dir, pdbid)):
                docked_ligands.append([('',None)])
                continue
            positives = [pdb for pdb 
                        in os.listdir(os.path.join(data_dir, pdbid))
                        if 'actives' in pdb]
            positives = list(zip(positives, [1] * len(positives)))
            negatives = [pdb for pdb 
                        in os.listdir(os.path.join(data_dir, pdbid))
                        if 'decoys' in pdb]
            
            negatives = list(zip(negatives, [0] * len(negatives)))

            docked_ligands.append((positives + negatives))

        data['ligands'] = docked_ligands
        data = data.explode('ligands').reset_index(drop=True)
        data[['ligands', 'label']] = pd.DataFrame(
            data['ligands'].tolist(), index=data.index)

        return data[data.label.notna()]



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
"""
def accuracy(output, label):
    pred = output.unsqueeze(0).max(1)[1].type_as(label)
    correct = pred.eq(label).double()
    return correct.sum() / len(label)

"""
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)