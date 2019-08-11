import argparse
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
                help="Dataset string ('aifb', 'mutag', 'bgs', 'am')")
ap.add_argument("-e", "--epochs", type=int, default=50,
                help="Number training epochs")
ap.add_argument("-hd", "--hidden", type=int, default=16,
                help="Number hidden units")
ap.add_argument("-do", "--dropout", type=float, default=0.,
                help="Dropout rate")
ap.add_argument("-b", "--bases", type=int, default=-1,
                help="Number of bases used (-1: all)")
ap.add_argument("-lr", "--learnrate", type=float, default=0.01,
                help="Learning rate")
ap.add_argument("-l2", "--l2norm", type=float, default=0.,
                help="L2 normalization of input weights")
ap.add_argument('--no-cuda', action='store_true', default=False, 
                help='Disables CUDA training.')
fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('--testing', dest='validation', action='store_false')
ap.set_defaults(validation=True)

args = vars(ap.parse_args())
print(args)

DATASET = args['dataset']
NB_EPOCH = args['epochs']
LR = args['learnrate']
L2 = args['l2norm']
HIDDEN = args['hidden']
BASES = args['bases']
DO = args['dropout']
USE_CUDA = not args['no_cuda'] and torch.cuda.is_available()

if USE_CUDA:
    torch.cuda.manual_seed(0)

start_time = time.time()
dirname = os.path.dirname(os.path.realpath(sys.argv[0]))

with open(dirname + '/' + DATASET + '.pickle', 'rb') as f:
    data = pickle.load(f)

A = data['A']
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
y = torch.tensor(y)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

if USE_CUDA:
    y = y.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

class GraphClassifier(nn.Module):
    
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_bases, dropout, support):
        super(GraphClassifier, self).__init__()
        self.gcn_1 = GraphConvolution(input_dim, hidden_dim, num_bases=num_bases, activation="relu",
                                      support=support)
        self.gcn_2 = GraphConvolution(hidden_dim, output_dim, num_bases=num_bases, activation="softmax",
                                     featureless=False, support=support)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs, mask=None):
        output = self.gcn_1(inputs, mask=mask)
        output = self.dropout(output)
        output = self.gcn_2([output]+inputs[1:], mask=mask)
        return output

if __name__ == "__main__":
    model = GraphClassifier(A[0].shape[0], HIDDEN, output_dimension, BASES, DO, len(A))
    model.to(device)
    if USE_CUDA:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2)
    criterion = nn.CrossEntropyLoss()
    X = sparse.csr_matrix(A[0].shape).todense()
    for epoch in range(NB_EPOCH):
        t = time.time()
        output = model([X]+A)

        # loss = criterion(output[idx_train], gold)
        loss = multi_labels_nll_loss(output[idx_train], y[idx_train])

        # score = accuracy_score(output[idx_train].argmax(dim=-1), gold)
        score = accuracy(output[idx_train], y[idx_train], USE_CUDA)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # test_score = accuracy_score(test_output.argmax(dim=-1), test_gold)
        val_score = accuracy(output[idx_val], y[idx_val], USE_CUDA)

        # test_loss = criterion(test_output, test_gold)
        val_loss = multi_labels_nll_loss(output[idx_val], y[idx_val])

        print("Epoch: {:04d}".format(epoch+1),
                "train_accuracy: {:.4f}".format(score),
                "train_loss: {:.4f}".format(loss.item()),
                "val_accuracy: {:.4f}".format(val_score),
                "val_loss: {:.4f}".format(val_loss.item()),
                "time: {:.4f}".format(time.time() - t))

    model.eval()
    output = model([X]+A)
    test_score = accuracy(output[idx_test], y[idx_test], USE_CUDA)
    test_loss = multi_labels_nll_loss(output[idx_test], y[idx_test])
    print("test_accuracy: {:.4f}".format(test_score),
          "loss: {:.4f}".format(test_loss.item()),
          "total time: {:.4f}".format(time.time() - start_time))
