from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from data_utils import PDBBindDataset, accuracy
from models import Model
from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
dataset = PDBBindDataset(num_positive=args.num_positive,
                         num_negative=args.num_negative)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(args.train_test_split * dataset_size))
np.random.shuffle(indices)

test_indices = indices[:split]
test_sampler = SubsetRandomSampler(test_indices)
test_loader = DataLoader(dataset, 
                         #batch_size=args.batch_size,
                         batch_size=1,
                         sampler=test_sampler)

model = Model(n_out=dataset.__nlabels__(),
              n_feat=dataset.__nfeats__(), 
              n_attns=args.n_attns, 
              n_dense=args.n_dense,
              dim_attn=args.dim_attn,
              dim_dense=args.dim_dense,
              dropout=args.dropout)


def compute_test():
    model.eval()

    losses_batch = []
    acc_batch = []

    for _ in range(len(test_loader)):
        try:
            (X, A, D), label = next(iter(test_loader))

            if args.cuda:
                X = X.cuda()
                A = A.cuda()
                D = D.cuda()
                label = label.cuda()

            output = model(X=X.squeeze(), 
                        A=A.squeeze(), 
                        D=D.squeeze())
            loss_test = F.nll_loss(output.unsqueeze(0), label.long())
            acc_test = accuracy(output, label)
            
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