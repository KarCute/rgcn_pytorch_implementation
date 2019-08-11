from collections import Counter

import numpy as np
import torch


def get_splits(y, train_idx, val_idx, test_idx, validation=True):
    # Make dataset splits
    # np.random.shuffle(train_idx)
    if validation:
        idx_train = train_idx
        idx_val = val_idx
        idx_test = test_idx
    else:
        idx_train = train_idx
        idx_val = train_idx  # no validation
        idx_test = test_idx

    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)

    y_train[idx_train] = np.array(y[idx_train].todense())
    y_val[idx_val] = np.array(y[idx_val].todense())
    y_test[idx_test] = np.array(y[idx_test].todense())

    return y_train, y_val, y_test, idx_train, idx_val, idx_test

def accuracy(output, labels, is_cuda):
    output = np.array(output.detach())
    cnt = len(np.where(labels)[1])
    counter = Counter(np.where(labels)[0])
    preds = np.zeros_like(labels)
    for idx in range(labels.shape[0]):
        labels_1_length = counter[idx]
        predict_1_index = np.argsort(-output[idx])[:labels_1_length]
        preds[idx][predict_1_index] = 1
    preds = torch.FloatTensor(preds)
    if is_cuda:
        preds = preds.cuda()
    correct = preds.type_as(labels).mul(labels).sum()
    return correct.item() / cnt, preds

def multi_labels_nll_loss(output, labels):
    # labels和output按位点乘，结果相加，除以labels中1的总数，作为适用于多标签的nll_loss。
    loss = labels.type_as(output).mul(output).sum()
    cnt = len(np.where(labels)[1])
    return loss / cnt
