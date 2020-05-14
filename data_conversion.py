import numpy as np
import scipy.sparse as sp
import torch

def encode_onehot(labels):
    classes = set()
    for label in labels:
        classes |= set(label)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array([np.sum([classes_dict.get(l) for l in label], axis=0) for label in labels], dtype=np.int32)
    return labels_onehot, len(classes)

# load dataset for rgcn
# {adjacencies, features, labels, labeled_nodes_idx, train_idx, test_idx, relations_dict, train_names, test_names}
def load_data_fb(path, dataset):
    print('Loading {} dataset...'.format(dataset))
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))

    # Get features
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    features = sp.csr_matrix(features)
    # print('features: ', features)

    # Get labels and labeled_nodes_idx
    labels = list(map(lambda x: x.split(','), idx_features_labels[:, -1]))
    labels, nclass = encode_onehot(labels)
    labels = sp.csr_matrix(labels)
    labeled_nodes_idx = list(labels.nonzero()[0])
    # print('labels: ', labels)

    # Get relations_dict(temporarily unknown)
    rel_dict = {}
    idx_rel = np.genfromtxt("{}{}.rel".format(path, dataset), dtype=np.dtype(str))
    for index in range(len(idx_rel)):
        e1 = idx_rel[:, 0]
        rel_dict[e1[index]] = index
    # print('rel_dict: ', rel_dict)

    # my method get classes adjancencies
    value = []
    row = []
    col = []
    for i in range(len(idx_rel)):
        value.append([])
        row.append([])
        col.append([])
    adjacencies = []
    if dataset == 'cora':
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    else:
        idx = np.array(idx_features_labels[:, 1], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    # 将每组关系中的entity用索引表示
    edges = np.array(list(map(idx_map.get, edges_unordered[:, :2].flatten())), dtype=np.int32).reshape(edges_unordered[:, :2].shape)
    e1 = np.array(edges[:, 0], dtype=np.int32)
    e2 = np.array(edges[:, 1], dtype=np.int32)
    relation = np.array(edges_unordered[:, -1],dtype=np.int32)
    amount = len(relation)
    for index in range(amount):
        value[relation[index]].append(1)
        row[relation[index]].append(e1[index])
        col[relation[index]].append(e2[index])
    for index in range(len(idx_rel)):
        adjacencies.append(sp.coo_matrix((value[index],(row[index],col[index])), shape=(labels.shape[0],labels.shape[0]), dtype=np.float32))
    print("adj1: {}".format(adjacencies[1]))
    print("adj2: {}".format(adjacencies[2]))
    print("adj3: {}".format(adjacencies[3]))

    # Divide train_idx and test_idx
    train_idx = range(len(idx_map) // 10 * 8)
    val_idx = range(len(idx_map) // 10 * 8, len(idx_map) // 10 * 9)
    test_idx = range(len(idx_map) // 10 * 9, len(idx_map))
    name = np.array(idx_features_labels[:, 0],dtype=str)
    train_name = name[0 : len(name) // 10 * 8]
    val_name = name[len(name) // 10 * 8 : len(name) // 10 * 9]
    test_name = name[len(name) // 10 * 9 : len(name)]
    print('train_idx: ',train_idx)
    print('val_idx: ', val_idx)
    print('test_idx: ', test_idx)
    print('train_name: ', train_name)
    print('val_name: ', val_name)
    print('test_name: ', test_name)


    print('Loading {} dataset finishes...'.format(dataset))
    return adjacencies,features,labels,labeled_nodes_idx,train_idx,test_idx,val_idx,rel_dict,train_name,test_name,val_name

def load_data_cora(path,dataset):
    print('Loading {} dataset...'.format(dataset))
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))

    # Get features
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    features = sp.csr_matrix(features)
    print('features: ', features)

    # Get labels and labeled_nodes_idx
    labels = list(map(lambda x: x.split(','), idx_features_labels[:, -1]))
    labels, nclass = encode_onehot(labels)
    labels = sp.csr_matrix(labels)
    labeled_nodes_idx = list(labels.nonzero()[0])
    print('labels: ', labels)

    # Get relations_dict(temporarily unknown)
    rel_dict = {'0':0}

    # my method get classes adjancencies
    adjacencies = []
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    # 将每组关系中的entity用索引表示
    edges = np.array(list(map(idx_map.get, edges_unordered[:, :2].flatten())), dtype=np.int32).reshape(edges_unordered[:, :2].shape)
    # 构建图的邻接矩阵，用坐标形式的稀疏矩阵表示，非对称邻接矩阵
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adjacencies.append(adj)
    print('adj: ', adjacencies[0])


    # Divide train_idx and test_idx
    train_idx = range(len(idx_map) // 10 * 8)
    val_idx = range(len(idx_map) // 10 * 8, len(idx_map) // 10 * 9)
    test_idx = range(len(idx_map) // 10 * 9, len(idx_map))
    name = np.array(idx_features_labels[:, 0],dtype=str)
    train_name = name[0 : len(name) // 10 * 8]
    val_name = name[len(name) // 10 * 8 : len(name) // 10 * 9]
    test_name = name[len(name) // 10 * 9 : len(name)]
    print('train_idx: ',train_idx)
    print('val_idx: ', val_idx)
    print('test_idx: ', test_idx)
    print('train_name: ', train_name)
    print('val_name: ', val_name)
    print('test_name: ', test_name)

    print('Loading {} dataset finishes...'.format(dataset))
    return adjacencies,features,labels,labeled_nodes_idx,train_idx,test_idx,val_idx,rel_dict,train_name,test_name,val_name