from __future__ import print_function
from data_conversion import *
from utils import *

import pickle as pkl

import os
import sys
import time
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="aifb",
                help="Dataset string ('aifb', 'mutag', 'bgs', 'am')")

args = vars(ap.parse_args())

print(args)

# Define parameters
DATASET = args['dataset']

NUM_GC_LAYERS = 2  # Number of graph convolutional layers

# Get data
# if DATASET == 'aifb' or DATASET == 'am' or DATASET == 'bgs' or DATASET == 'mutag':
#     A, X, y, labeled_nodes_idx, train_idx, test_idx, rel_dict, train_names, test_names = load_data(
#     DATASET)
if DATASET == 'cora':
    A, X, y, labeled_nodes_idx, train_idx, test_idx, val_idx, rel_dict, train_names, test_names, val_names = load_data_cora('../pyGAT/data/' + DATASET + '/',
    DATASET)
else:
    A, X, y, labeled_nodes_idx, train_idx, test_idx, val_idx, rel_dict, train_names, test_names, val_names = load_data_other('../pyGAT/data/' + DATASET + '/',
    DATASET)

# rel_list = list(range(len(A)))
# for key, value in rel_dict.items():
#     if value * 2 >= len(A):
#         continue
#     rel_list[value * 2] = key
#     rel_list[value * 2 + 1] = key + '_INV'


num_nodes = A[0].shape[0] # 8285
A.append(sp.identity(A[0].shape[0]).tocsr())  # add identity matrix

support = len(A) # 91

print("Relations used and their frequencies" + str([a.sum() for a in A]))

print("Calculating level sets...")
t = time.time()
# Get level sets (used for memory optimization)
bfs_generator = bfs_relational(A, labeled_nodes_idx)
lvls = list()
lvls.append(set(labeled_nodes_idx))
lvls.append(set.union(*bfs_generator.__next__()))
print("Done! Elapsed time " + str(time.time() - t))

# Delete unnecessary rows in adjacencies for memory efficiency
todel = list(set(range(num_nodes)) - set.union(lvls[0], lvls[1]))
for i in range(len(A)):
    csr_zero_rows(A[i].tocsr(), todel)

data = {'A': A,
        'X': X,
        'y': y,
        'train_idx': train_idx,
        'test_idx': test_idx,
        'val_idx':val_idx
        }

dirname = os.path.dirname(os.path.realpath(sys.argv[0]))

with open(dirname + '/' + DATASET + '.pickle', 'wb') as f:
    pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)
