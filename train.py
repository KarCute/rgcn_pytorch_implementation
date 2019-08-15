import argparse
import glob
import time
import sys
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import numpy as np
from collections import Counter
from scipy import sparse
from sklearn.metrics import accuracy_score

from layers import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(torch.cuda.is_available())
np.random.seed()
torch.manual_seed(0)

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="FB15K237",
                help="Dataset string ('FB15K237', 'WN18RR', 'WN18RR_sub30000', 'cora')")
ap.add_argument("-bad", "--bad", type=int, default=100,
                help="bad counter")
ap.add_argument("-e", "--epochs", type=int, default=1000,
                help="Number training epochs")
ap.add_argument("-hd", "--hidden", type=int, default=16,
                help="Number hidden units")
ap.add_argument("-do", "--dropout", type=float, default=0.,
                help="Dropout rate")
ap.add_argument("-b", "--bases", type=int, default=-1,
                help="Number of bases used (-1: all)")
ap.add_argument("-lr", "--learnrate", type=float, default=0.01,
                help="Learning rate")
ap.add_argument("-l2", "--l2norm", type=float, default=0.6,
                help="L2 normalization of input weights")
ap.add_argument('--experiment', type=str, default='GAT',
                help='Name of current experiment.')
ap.add_argument('--no-cuda', action='store_true', default=False, 
                help='Disables CUDA training.')
fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('--testing', dest='validation', action='store_false')
ap.set_defaults(validation=True)

args = vars(ap.parse_args())
print(args)

DATASET = args['dataset']
BAD = args['bad']
NB_EPOCH = args['epochs']
LR = args['learnrate']
L2 = args['l2norm']
HIDDEN = args['hidden']
BASES = args['bases']
DO = args['dropout']
EXP = args['experiment']
USE_CUDA = not args['no_cuda'] and torch.cuda.is_available()

if USE_CUDA:
    torch.cuda.manual_seed(0)

start_time = time.time()
dirname = os.path.dirname(os.path.realpath(sys.argv[0]))

with open(dirname + '/' + DATASET + '.pickle', 'rb') as f:
    data = pickle.load(f)

A = data['A']
X = np.array(data['X'].todense())
#y = data['y']
y = np.array(data['y'].todense())
idx_train = data['train_idx']
idx_val = data['val_idx']
idx_test = data['test_idx']
del data

for i in range(len(A)):
    d = np.array(A[i].sum(1)).flatten()
    d_inv = 1. / d
    d_inv[np.isinf(d_inv)] = 0.
    D_inv = sparse.diags(d_inv)
    A[i] = D_inv.dot(A[i]).tocsr()

A = [i for i in A if len(i.nonzero()[0]) > 0]



output_dimension = y.shape[1]
support = len(A)
y = torch.FloatTensor(y)
X = torch.FloatTensor(X)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

if USE_CUDA:
    y = y.cuda()
    X = X.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

class GraphClassifier(nn.Module):
    
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_bases, dropout, support):
        super(GraphClassifier, self).__init__()
        self.gcn_1 = GraphConvolution(input_dim, hidden_dim, num_bases=num_bases, activation="relu",
                                      featureless=False, support=support)
        self.gcn_2 = GraphConvolution(hidden_dim, output_dim, num_bases=num_bases, activation="softmax",
                                     featureless=False, support=support)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs, mask=None):
        output = self.gcn_1(inputs, mask=mask)
        output = self.dropout(output)
        output = self.gcn_2([output]+inputs[1:], mask=mask)
        return output

if __name__ == "__main__":
    model = GraphClassifier(X.shape[1], HIDDEN, output_dimension, BASES, DO, len(A))
    model.to(device)
    if USE_CUDA:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2)
    criterion = torch.nn.BCEWithLogitsLoss(size_average=True)
    #criterion = nn.CrossEntropyLoss()
    #X = sparse.csr_matrix(X.shape).todense()
    loss_values = []
    best = NB_EPOCH + 1
    best_epoch = 0
    bad_counter = 0

    files = glob.glob('./{}/*.pkl'.format(EXP))
    for file in files:
        os.remove(file)
    if not os.path.exists(EXP):
        os.mkdir('{}'.format(EXP))

    for epoch in range(NB_EPOCH):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model([X]+A)

        output = F.log_softmax(output)
        #loss = criterion(output[idx_train], y[idx_train])
        loss = multi_labels_nll_loss(output[idx_train], y[idx_train])
        #score = accuracy_score(output[idx_train].argmax(dim=-1), y[idx_train].argmax(dim=-1))
        score = accuracy(output[idx_train], y[idx_train], USE_CUDA)

        loss.backward()
        optimizer.step()

        model.eval()
        output = model([X]+A)
        output = F.log_softmax(output)
        #val_score = accuracy_score(output[idx_val].argmax(dim=-1), y[idx_val].argmax(dim=-1))
        val_score = accuracy(output[idx_val], y[idx_val], USE_CUDA)
        #val_loss = criterion(output[idx_val], y[idx_val])
        val_loss = multi_labels_nll_loss(output[idx_val], y[idx_val])

        print("Epoch: {:04d}".format(epoch+1),
                "train_accuracy: {:.4f}".format(score),
                "train_loss: {:.4f}".format(loss.item()),
                "val_accuracy: {:.4f}".format(val_score),
                "val_loss: {:.4f}".format(val_loss.item()),
                "time: {:.4f}".format(time.time() - t))

        loss_values.append(val_loss)
        torch.save(model.state_dict(), './{}/{}.pkl'.format(EXP, epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        # 损失连续100次迭代没有优化时，则提取停止
        if bad_counter == BAD:
            break
        files = glob.glob('./{}/*.pkl'.format(EXP))
        for file in files:
            epoch_nb = int(file.split('/')[-1].split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

    files = glob.glob('./{}/*.pkl'.format(EXP))
    for file in files:
        epoch_nb = int(file.split('/')[-1].split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('./{}/{}.pkl'.format(EXP, best_epoch)))

    model.eval()
    output = model([X]+A)
    output = F.log_softmax(output)
    #test_score = accuracy_score(output[idx_test].argmax(dim=-1), y[idx_test].argmax(dim=-1))
    #test_loss = criterion(output[idx_test], y[idx_test])
    test_score = accuracy(output[idx_test], y[idx_test], USE_CUDA)
    test_loss = multi_labels_nll_loss(output[idx_test], y[idx_test])
    print("test_accuracy: {:.4f}".format(test_score),
          "test_loss: {:.4f}".format(test_loss.item()),
          "total time: {:.4f}".format(time.time() - start_time))
