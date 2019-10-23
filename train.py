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
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_utils import PDBBindDataset, accuracy
from models import Model
from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

writer = SummaryWriter(log_dir = args.log_dir)

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
train_indices, test_indices, val_indices \
    = indices[2*split:], indices[:split], indices[split:2*split]

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

model = Model(n_out=dataset.__nlabels__(),
              n_feat=dataset.__nfeats__(), 
              n_attns=args.n_attns, 
              n_dense=args.n_dense,
              dim_attn=args.dim_attn,
              dim_dense=args.dim_dense,
              dropout=args.dropout)
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    losses_batch = []
    acc_batch = []
    for _ in range(args.batch_size): # not really "the batch"
        try:
            (X, A, D), label = next(iter(train_loader))
            if args.cuda:
                X = X.cuda()
                A = A.cuda()
                D = D.cuda()
                label = label.cuda()

            output = model(X=X.squeeze(), 
                           A=A.squeeze(), 
                           D=D.squeeze())
            loss_train = F.nll_loss(output.unsqueeze(0), label.long())
            acc_train = accuracy(output, label)

            losses_batch.append(loss_train)
            acc_batch.append(acc_train)

            loss_train.backward()
            optimizer.step()
        except BaseException as e:
            print(e)
            pass

    avg_loss = torch.mean(torch.Tensor(losses_batch))
    avg_acc = torch.mean(torch.Tensor(acc_batch))

    writer.add_scalar('Training Loss', avg_loss.data.items(), epoch)
    writer.add_scalar('Training Accuracy', avg_acc.data.items(), epoch)

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(avg_loss.data.item()),
          'acc_train: {:.4f}'.format(avg_acc.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return avg_loss.data.item()

def evaluate(epoch):
    model.eval()

    losses_batch = []
    acc_batch = []
    for _ in range(args.batch_size):
        try:
            (X, A, D), label = next(iter(val_loader))

            if args.cuda:
                X = X.cuda()
                A = A.cuda()
                D = D.cuda()
                label = label.cuda()

            output = model(X=X.squeeze(), 
                        A=A.squeeze(), 
                        D=D.squeeze())
            loss_val = F.nll_loss(output.unsqueeze(0), label.long())
            acc_val = accuracy(output, label)

            losses_batch.append(loss_val)
            acc_batch.append(acc_val)
        except BaseException as e:
            print(e)
        
    avg_loss = torch.mean(torch.Tensor(losses_batch))
    avg_acc = torch.mean(torch.Tensor(acc_batch))

    writer.add_scalar('Validation Loss', avg_loss.data.items(), epoch)
    writer.add_scalar('Validation Accuracy', avg_acc.data.items(), epoch)

    print("Validation set results:",
          "loss= {:.4f}".format(avg_loss.data),
          "accuracy= {:.4f}".format(avg_acc.data))


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

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    if epoch % 20 == 0:
        evaluate(epoch)
    torch.save(model.state_dict(), '{}.pkl'.format(epoch))

    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()