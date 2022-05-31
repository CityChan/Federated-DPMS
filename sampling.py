
import numpy as np
import torch
import scipy
from torch.utils.data import Dataset
import torch
import copy
from torchvision import datasets, transforms

def Dataset_IIDsampling(dataset,num_users):
    """
    dataset:  Input the created dataset(train or validation or test)
    num_usersL number of local models
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        # randomly pick the index and store it in a dictionary with a key
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def Dataset_non_IIDsampling(dataset,num_users):
    """
    dataset:  Input the created dataset(train or validation or test)
    num_usersL number of local models
    """
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    half_users = int(num_users/2)
    for i in range(half_users):
        low_num = int(1.5*len(dataset)/num_users)
        high_num = int(2*len(dataset)/num_users)
        num_items = np.random.randint(low = low_num, high=high_num, size=1)
        # randomly pick the index and store it in a dictionary with a key
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    num_items = int(len(all_idxs)/(num_users - half_users))
    for i in range(half_users,num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])    
    return dict_users


class LocalDataset(Dataset):
    """
    because torch.dataloader need override __getitem__() to iterate by index
    this class is map the index to local dataloader into the whole dataloader
    """
    def __init__(self, dataset, Dict):
        self.dataset = dataset
        self.idxs = [int(i) for i in Dict]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        X, y = self.dataset[self.idxs[item]]
        return X, y
    
def LocalDataloaders(dataset, dict_users, batch_size, ShuffleorNot = True, BatchorNot = True):
    """
    dataset: the same dataset object
    dict_users: dictionary of index of each local model
    batch_size: batch size for each dataloader
    ShuffleorNot: Shuffle or Not
    BatchorNot: if False, the dataloader will give the full length of data instead of a batch, for testing
    """
    num_users = len(dict_users)
    loaders = []
    for i in range(num_users):
        if BatchorNot== True:
            loader = torch.utils.data.DataLoader(
                        LocalDataset(dataset,dict_users[i]),
                        batch_size=batch_size,
                        shuffle = ShuffleorNot,
                        num_workers=0)
        else:
            loader = torch.utils.data.DataLoader(
                        LocalDataset(dataset,dict_users[i]),
                        batch_size=len(LocalDataset(dataset,dict_users[i])),
                        shuffle = ShuffleorNot,
                        num_workers=0)
        loaders.append(loader)
    return loaders

def LocalDataloaders_sample(dataset, dict_users, batch_size, ShuffleorNot = True, BatchorNot = True, mini = 0.1):
    """
    dataset: the same dataset object
    dict_users: dictionary of index of each local model
    batch_size: batch size for each dataloader
    ShuffleorNot: Shuffle or Not
    BatchorNot: if False, the dataloader will give the full length of data instead of a batch, for testing
    """
    num_users = len(dict_users)
    loaders = []
    for i in range(num_users):
        num_data = len(dict_users[i])
        mini_num_data = int(mini*num_data)
        whole_range = range(num_data)
        mini_range = np.random.choice(whole_range, mini_num_data)
        mini_dict_users = [dict_users[i][j] for j in mini_range]
        if BatchorNot== True:
            loader = torch.utils.data.DataLoader(
                        LocalDataset(dataset,mini_dict_users),
                        batch_size=batch_size,
                        shuffle = ShuffleorNot,
                        num_workers=0)
        else:
            loader = torch.utils.data.DataLoader(
                        LocalDataset(dataset,mini_dict_users),
                        batch_size=len(LocalDataset(dataset,dict_users[i])),
                        shuffle = ShuffleorNot,
                        num_workers=0)
        loaders.append(loader)
    return loaders


def average_weights(w):
    """
    average the weights from all local models
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def get_dataset_cifar10_extr_noniid(num_users, n_class, nsamples, rate_unbalance, rand_seed = 0, all_class = False, all_class_prop = 0.05):
    data_dir = '../data/cifar/'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                   transform=apply_transform)

    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

    # Chose euqal splits for every user
    user_groups_train, user_groups_test = cifar_extr_noniid(train_dataset, test_dataset, num_users, n_class, nsamples, rate_unbalance,rand_seed,  all_class = all_class, all_class_prop = all_class_prop)
    return train_dataset, test_dataset, user_groups_train, user_groups_test

def cifar_extr_noniid(train_dataset, test_dataset, num_users, n_class, num_samples, rate_unbalance, rand_seed = 0, all_class = False, all_class_prop = 0.05):
    num_shards_train, num_imgs_train = int(50000/num_samples), num_samples
    num_classes = 10
    np.random.seed(rand_seed)
    num_imgs_perc_test, num_imgs_test_total = 1000, 10000
    assert(n_class * num_users <= num_shards_train)
    assert(n_class <= num_classes)
    idx_class = [i for i in range(num_classes)]
    idx_shard = [i for i in range(num_shards_train)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards_train*num_imgs_train)
    # labels = dataset.train_labels.numpy()
    labels = np.array(train_dataset.targets)
    idxs_test = np.arange(num_imgs_test_total)
    labels_test = np.array(test_dataset.targets)
    #labels_test_raw = np.array(test_dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    #print(idxs_labels_test[1, :])

    

    
    # divide and assign
    for i in range(num_users):
        user_labels = np.array([])
        rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        unbalance_flag = 0
        for rand in rand_set:
            if unbalance_flag == 0:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
            else:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
            unbalance_flag = 1
        user_labels_set = set(user_labels)
       
        for label in user_labels_set:
            dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test[int(label)*num_imgs_perc_test:int(label+1)*num_imgs_perc_test]), axis=0)
        #print(set(labels_test_raw[dict_users_test[i].astype(int)]))
    if all_class == True:  
        #basic data for each class
        section = 5000/num_samples
        for idx in range(num_users):
            for i in range(num_classes):       
                position = np.random.randint(section)
                head = int((position + i*section)*num_imgs_train)
                tail = int(head + num_imgs_train*all_class_prop)
                dict_users_train[idx] = np.concatenate((dict_users_train[idx], idxs[head:tail]), axis=0)
                user_labels = np.concatenate((user_labels, labels[head:tail]), axis=0)

    return dict_users_train, dict_users_test


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts

def partition_data(n_users, alpha=0.5,rand_seed = 0):
    data_dir = '../data/cifar/'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                   transform=apply_transform)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)
    n_train = len(train_dataset)
    min_size = 0
    K = 10
    N = len(train_dataset)
    net_dataidx_map = {}
    np.random.seed(rand_seed)
    y_train = np.array(train_dataset.targets)
    while min_size < 10:
        idx_batch = [[] for _ in range(n_users)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_users))
            ## Balance
            proportions = np.array([p*(len(idx_j)<N/n_users) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_users):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return (train_dataset, test_dataset,net_dataidx_map, traindata_cls_counts)

def partition_data_FMNIST(n_users, alpha=0.5,rand_seed = 0):
    data_dir = '../data/FMNIST/'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))])
    train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)
    test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
    n_train = len(train_dataset)
    min_size = 0
    K = 100
    N = len(train_dataset)
    net_dataidx_map = {}
    np.random.seed(rand_seed)
    y_train = np.array(train_dataset.targets)
    while min_size < 10:
        idx_batch = [[] for _ in range(n_users)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_users))
            ## Balance
            proportions = np.array([p*(len(idx_j)<N/n_users) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_users):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return (train_dataset, test_dataset,net_dataidx_map, traindata_cls_counts)


def partition_data_cifar100(n_users, alpha=0.5,rand_seed = 0):
    data_dir = '../data/cifar/'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                   transform=apply_transform)
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=apply_transform)
    n_train = len(train_dataset)
    min_size = 0
    K = 100
    N = len(train_dataset)
    net_dataidx_map = {}
    np.random.seed(rand_seed)
    y_train = np.array(train_dataset.targets)
    while min_size < 10:
        idx_batch = [[] for _ in range(n_users)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_users))
            ## Balance
            proportions = np.array([p*(len(idx_j)<N/n_users) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_users):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return (train_dataset, test_dataset,net_dataidx_map, traindata_cls_counts)

