import torch
from torch.utils.data import Dataset
from utils import get_pdbbind_features, one_hot_encoding
import pandas as pd

class PDBBindDataset(Dataset):
    def __init__(self, 
                 index_csv='./data/pdbbind_binding_affinity.csv', 
                 data_dir='./data/pdbbind/v2018',
                 num_positive=1000,
                 num_negative=1000,
                 binding_type='ic50', 
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

    def __getitem__(self, index):
        pdbid = self.data.loc[index, 'id']
        try:
            X, A, D = get_pdbbind_features(pdbid, data_dir=self.data_dir)        
            # label = one_hot_encoding(self.data.loc[index, 'label'], {0, 1})
            label = self.data.loc[index, 'label']
            return (torch.Tensor(X), torch.Tensor(A), torch.Tensor(D)), label

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
"""





