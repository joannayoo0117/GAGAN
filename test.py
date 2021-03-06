from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from data_utils import PDBBindDataset, accuracy
from models import GAGAN, GAT
from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    try:
        torch.cuda.device(args.cuda_device)
    except BaseException as e:
        print(e)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
dataset = PDBBindDataset(num_positive=args.num_positive,
                         num_negative=args.num_negative,
                         seed=args.seed,
                         train_test_split=args.train_test_split)
train_indices, test_indices, val_indices = dataset.__train_test_split__()

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, 
                          #batch_size=args.batch_size,
                          batch_size=1, # Loads graph one at a time
                          sampler=train_sampler)
test_loader = DataLoader(dataset, 
                         #batch_size=args.batch_size,
                         batch_size=1,
                         sampler=test_sampler)
val_loader = DataLoader(dataset, 
                        #batch_size=args.batch_size,
                        batch_size=1,
                        sampler=val_sampler)

model = None
if args.model == 'GAGAN':
    model = GAGAN(n_out=dataset.__nlabels__(),
                  n_feat=dataset.__nfeats__(), 
                  n_attns=args.n_attns, 
                  n_dense=args.n_dense,
                  dim_attn=args.dim_attn,
                  dim_dense=args.dim_dense,
                  dropout=args.dropout)
elif args.model == 'GAT':
    model = GAT(nclass=dataset.__nlabels__(),
                nfeat=dataset.__nfeats__(), 
                nhid=args.dim_attn,
                nheads=args.n_attns,
                nhid_linear=args.dim_dense,
                nlinear=args.n_dense,
                alpha=args.alpha,
                dropout=args.dropout,
                use_distance_aware_adj=args.use_dist_aware_adj)
else:
    raise ValueError('args.model should be one of GAGAN, GAT')

criterion = nn.CrossEntropyLoss()

def compute_test():
    model.eval()

    losses_batch = []
    acc_batch = []

    for _ in range(len(test_loader)):
        try:
            (X, A, A2), label = next(iter(test_loader))

            if args.cuda:
                X = X.cuda()
                A = A.cuda()
                A2 = A2.cuda()
                label = label.cuda()

            output = model(X=X.squeeze(), 
                        A=A.squeeze(), 
                        A2=A2.squeeze())
            loss_test = criterion(output, label.view(-1))
            acc_test = accuracy(output, label.view(-1))
            
            losses_batch.append(loss_test)
            acc_batch.append(acc_test)
        except BaseException as e:
            print(e)
    
    avg_loss = torch.mean(torch.Tensor(losses_batch))
    avg_acc = torch.mean(torch.Tensor(acc_batch))

    print("Test set results:",
          "loss= {:.4f}".format(avg_loss.data),
          "accuracy= {:.4f}".format(avg_acc.data))


# Restore best model
model.load_state_dict(torch.load(args.model_file))

# Testing
compute_test()