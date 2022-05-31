from PIL import Image
from os.path import join
import imageio
from torch import nn
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import os,sys,os.path
import pandas as pd
import pickle
import pydicom
import collections
import pprint
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import warnings
import tarfile
import zipfile
import random
from skimage.io import imread, imsave
from skimage.transform import resize
import time
import math
from sklearn.cluster import KMeans
import scipy
import torch.optim as optim
import copy
from scipy.optimize import linear_sum_assignment as linear_assignment
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import hashlib
import argparse

# import models and data from code

# from option import args_parser
from utils import Accuracy,GenData,ConcatDataset,getLocalMeans_global,getLocalMeans, getLocalMean, reparameterize,save_decoded_image, LocalGenerate,DatasetSplit
from models import CIFAR_CNN,CIFAR_VAE,FMNIST_CNN,FMNIST_VAE
from sampling import LocalDataset, LocalDataloaders, average_weights,partition_data,partition_data_FMNIST, partition_data_cifar100, LocalDataloaders_sample,partition_data
from Localupdate import LocalUpdate





torch.set_default_dtype(torch.float64)
torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
device = torch.device(f'cuda:{4}' if torch.cuda.is_available() else 'cpu')
np.set_printoptions(threshold=np.inf)


parser = argparse.ArgumentParser()

#Data specifc paremeters
parser.add_argument('--dataset', type=str, default='CIFAR10',
                    help='CIFAR10, CIFAR100, FMNIST') 
#Training specifc parameters
parser.add_argument('--log_frq', type=int, default=5,
                    help='frequency of logging')
parser.add_argument('--batch_size', type=int, default=64,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=50,
                    help='number of epochs')         
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=10,
                    help='number of steps to drop the lr')
parser.add_argument('--dropout_rate', type=float, default=0.2,
                    help='drop out rate for training') 
parser.add_argument('--tag', type=str, default='centralized',
                    help='centralized, federated')
parser.add_argument('--num_users', type=int, default=10,
                    help='number of local models')
parser.add_argument('--update_frac', type=float, default=1,
                    help='frac of local models to update')
parser.add_argument('--local_ep', type=int, default=5,
                    help='iterations of local updating')
parser.add_argument('--beta', type=float, default=0.5,
                    help='beta for non-iid distribution')
parser.add_argument('--seed', type=int,default=0,
                    help='random seed for generating datasets')
parser.add_argument('--mini', type=float, default=1,
                    help='sample rate of each local data')
parser.add_argument('--moon_mu', type=int, default=5,
                    help='weight for moon term')
parser.add_argument('--moon_temp', type=float, default=0.5,
                    help='temperture for moon')
parser.add_argument('--prox_mu', type=float, default=0.001,
                    help='weight for prox term')
parser.add_argument('--pretrain', type=int, default=20,
                    help='pretrain epochs for vae')
parser.add_argument('--gen_num', type=int, default=50,
                    help='number of generating images')
parser.add_argument('--std', type=int, default=4,
                    help='std for generating means')
parser.add_argument('--inputsize', type=int, default=32,
                    help='input size')
parser.add_argument('--inputchannel', type=int, default=3,
                    help='input channel')
parser.add_argument('--classes', type=int, default=10,
                    help='number of classes')
parser.add_argument('--code_len', type=int,default=32,
                    help='length of code')
parser.add_argument('--alg', type=str, default='FedAvg',
                    help='FedAvg, FedProx, Moon, FedVAE, DPMS, FedMix')
parser.add_argument('--eval_only',action="store_true", default=False,help='evaluate the model')

parser.add_argument('--vae_mu',type=float, default=0.05,
                    help='parameter for vae term')
parser.add_argument('--fedmix_lam',type=float, default=0.05,
                    help='parameter for fedmix lambda')
args = parser.parse_args()
# args = args_parser()
args.tag = 'federated'
args_hash = ''
for k,v in vars(args).items():
    if k == 'eval_only':
        continue
    args_hash += str(k)+str(v)
    
args_hash = hashlib.sha256(args_hash.encode()).hexdigest()

if args.dataset == 'CIFAR10':
    train_dataset,testset, dict_users, traindata_cls_counts = partition_data(n_users = args.num_users, alpha=args.beta,rand_seed = args.seed)
    args.inputsize = 32
    args.inputchannel = 3
    args.classes = 10
    
elif args.dataset == 'CIFAR100':
    train_dataset,testset, dict_users, traindata_cls_counts = partition_data_cifar100(n_users = args.num_users, alpha=args.beta,rand_seed = args.seed)
    args.inputsize = 32
    args.inputchannel = 3
    args.classes = 100
    
elif args.dataset == 'FMNIST':
    train_dataset,testset, dict_users, traindata_cls_counts = partition_data_FMNIST(n_users = args.num_users, alpha=args.beta,rand_seed = args.seed)
    args.inputsize = 28
    args.inputchannel = 1
    args.classes = 10


    
Loaders_train = LocalDataloaders_sample(train_dataset,dict_users,args.batch_size,ShuffleorNot = True, mini=args.mini)
Major_need_classes = []
Major_classes = []
Total_classes = []
if args.dataset == 'CIFAR10' or args.dataset == 'FMNIST':
    for idx in range(args.num_users):
        counts = [0]*10
        for batch_idx,(X,y) in enumerate(Loaders_train[idx]):
            batch = len(y)
            y = np.array(y)
            for i in range(batch):
                counts[int(y[i])] += 1
        counts = np.array(counts)
        major_need_classes = counts.argsort()[::-1][7:]
        major_classes = counts.argsort()[::-1][0:3]
        Total_classes.append(counts/np.sum(counts))
        Major_need_classes.append(major_need_classes)
        Major_classes.append(major_classes)
        
if args.dataset == 'CIFAR100':
    for idx in range(args.num_users):
        counts = [0]*100
        for batch_idx,(X,y) in enumerate(Loaders_train[idx]):
            batch = len(y)
            y = np.array(y)
            for i in range(batch):
                counts[int(y[i])] += 1
        counts = np.array(counts)
        major_need_classes = counts.argsort()[::-1][90:]
        major_classes = counts.argsort()[::-1][0:10]
        Total_classes.append(counts/np.sum(counts))
        Major_need_classes.append(major_need_classes)
        Major_classes.append(major_classes)

if args.alg == 'FedMix':
    images_means, labels_means = torch.Tensor().to(device), torch.Tensor().to(device)
    for idx in range(args.num_users):
        local_gen = LocalGenerate(args=args, dataset=train_dataset, idxs=dict_users[idx],device = device)
        images_mean, labels_mean = local_gen.generate()
        images_means = torch.cat([images_means, images_mean], dim=0)
        labels_means = torch.cat([labels_means, labels_mean], dim=0)
        
L2D = np.zeros((args.num_users,args.num_users))
for i in range(args.num_users):
    for j in range(args.num_users):
        common = np.intersect1d(Major_need_classes[i], Major_classes[j])
        L2D[i,j] = len(common)


if args.alg == 'FedAvg' or args.alg == 'FedProx' or args.alg == 'Moon' or args.alg == 'FedMix':  
    if args.dataset == 'FMNIST':
        global_model = FMNIST_CNN(in_channels=args.inputchannel,input_shape=[args.inputsize,args.inputsize],code_length=args.code_len, classes = args.classes,drop_rate=args.dropout_rate)
    
    else:
        global_model = CIFAR_CNN(in_channels=args.inputchannel,input_shape=[args.inputsize,args.inputsize],code_length=args.code_len, classes = args.classes,drop_rate=args.dropout_rate)
    
elif args.alg == 'FedVAE' or args.alg == 'DPMS':
    if args.dataset == 'FMNIST':
        global_model = FMNIST_VAE(in_channels=args.inputchannel,input_shape=[args.inputsize,args.inputsize],code_length=args.code_len, classes = args.classes,drop_rate=args.dropout_rate,decoder = True)  
    else:
        global_model = CIFAR_VAE(in_channels=args.inputchannel,input_shape=[args.inputsize,args.inputsize],code_length=args.code_len, classes = args.classes,drop_rate=args.dropout_rate,decoder = True)
print('# model parameters:', sum(param.numel() for param in global_model.parameters()))

# use multi-GPU to train parallelly
global_model = nn.DataParallel(global_model, device_ids = [4,5,6,7])
global_model.to(device)
for m in global_model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

        
logger = SummaryWriter('./logs')
checkpoint_dir = './checkpoint/'+args.tag+'/'+args.dataset+'/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# training:
    
global_weights = global_model.state_dict()

loader_test = torch.utils.data.DataLoader(testset, batch_size=64,shuffle=True, num_workers=2)
ACC = []
metrics = {'accuracy':ACC,  'max_accuracy':0, 'time':0}
start_time = time.time()
print(args)
if not args.eval_only:
    if args.alg == 'DPMS':
        # pre-training
        Local_models = []
        for idx in range(args.num_users):
            Local_models.append(copy.deepcopy(global_model))
            
        for epoch in tqdm(range(args.pretrain)):
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {epoch+1} |\n')
            global_model.train()
            m = max(int(args.update_frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            for idx in idxs_users:
                local_model = LocalUpdate(args,Loaders_train[idx], idxs=dict_users[idx], logger=logger)
                w, loss = local_model.update_weights(
                    model=Local_models[idx], global_round=epoch, u=args.vae_mu, device=device)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
#                 Local_Models[idx].load_state_dict(local_weights)
            
            # update global weights
            global_weights = average_weights(local_weights)
            global_model.load_state_dict(global_weights)
            Encoder = {}
            Classifier = {}
            for key,param in global_weights.items():
                if 'Encoder' in key:
                    Encoder[key] = param
                if 'Classifier' in key or 'Mu' in key or 'Var' in key: 
                    Classifier[key] = param

            for idx in range(args.num_users):
                local_Encoder = Encoder.copy()
                local_Classifier = Classifier.copy()
                local_Decoder = {}
                for key,param in Local_models[idx].state_dict().items():
                    if 'Decoder' in key: 
                        local_Decoder[key] = param 
                local_Encoder.update(local_Decoder)
                local_Encoder.update(local_Classifier)
                Local_models[idx].load_state_dict(local_Encoder)


            accuracy = 0
            count = 0
            global_model.eval()
            for cnt, (X,y) in enumerate(loader_test):
                X = X.to(device)
                m = torch.nn.Sigmoid()
                X = m(X)
                y = y.double().to(device)
                mu,logVar,p,_X = global_model(X)
                y_pred = p.argmax(1)
                accuracy += Accuracy(y,y_pred)
                count += 1
            print("accuracy of test at this round:",accuracy/count)
            ACC.append(accuracy/count)
            
        
        Models = []
        for idx in range(args.num_users):
            model = copy.deepcopy(global_model)
            model.load_state_dict(local_weights[idx])
            model.to(device)
            Models.append(model)

        print("Generate new dataset...")
        Loaders_train = LocalDataloaders_sample(train_dataset,dict_users,args.batch_size,ShuffleorNot = True, mini=args.mini)
        Means,Vars = getLocalMeans(copy.deepcopy(Models),Loaders_train,args.num_users,Major_classes,device)
#         Means,Vars = getLocalMeans_global(copy.deepcopy(global_model),Loaders_train,args.num_users,Major_classes,device)

        loaders_gen = []
        for idx in range(args.num_users):
            match_index = np.argmax(L2D[idx])
            gen_num = args.gen_num
            std = args.std
            dataset_gen = GenData(args, copy.deepcopy(global_model),copy.deepcopy(Models[match_index]),Means[match_index],Vars[match_index],Major_classes[match_index],gen_num,std,device,idx)
#             dataset_gen = GenData(args, copy.deepcopy(global_model),copy.deepcopy(global_model),Means[match_index],Vars[match_index],Major_classes[match_index],gen_num,std,device,idx)
            loader_gen = torch.utils.data.DataLoader(
                            dataset_gen,
                            batch_size=args.batch_size,
                            shuffle =True)
            loaders_gen.append(loader_gen)          
        print("Generation done...")

        Merge_loaders = []
        for idx in range(args.num_users):
            merge_dataset = ConcatDataset(Loaders_train[idx],loaders_gen[idx],device)
            merge_loader = torch.utils.data.DataLoader(merge_dataset, batch_size=args.batch_size,shuffle=True)
            Merge_loaders.append(merge_loader)


        global_model.module.setDecoder(False)
        for param in global_model.module.Decoder.parameters():
            param.requires_grad = False
        # training
        for epoch in tqdm(range(args.pretrain,args.num_epochs)):
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {epoch+1} |\n')
            global_model.train()
            m = max(int(args.update_frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            for idx in idxs_users:
                local_model = LocalUpdate(args, Merge_loaders[idx], idxs=dict_users[idx], logger=logger)
                w, loss = local_model.update_weights_VAEtoCNN(
                    model=copy.deepcopy(global_model), global_round=epoch, u=args.vae_mu, device=device)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            global_weights = average_weights(local_weights)

            # update global weights
            global_model.load_state_dict(global_weights)


            accuracy = 0
            count = 0
            global_model.eval()
            for cnt, (X,y) in enumerate(loader_test):
                X = X.to(device)
                m = torch.nn.Sigmoid()
                X = m(X)
                y = y.double().to(device)
                p = global_model(X)
                y_pred = p.argmax(1)
                accuracy += Accuracy(y,y_pred)
                count += 1
            print("accuracy of test at this round:",accuracy/count)
            ACC.append(accuracy/count)

    else:
        Local_Weights = []
        for epoch in tqdm(range(args.num_epochs)):
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {epoch+1} |\n')
            global_model.train()
            m = max(int(args.update_frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            for idx in idxs_users:
                local_model = LocalUpdate(args,Loaders_train[idx], idxs=dict_users[idx], logger=logger)
                if args.alg == 'FedAvg':
                    w, loss = local_model.update_weights_CNN(model=copy.deepcopy(global_model), global_round=epoch, device=device)
                if args.alg == 'FedProx':
                    w, loss = local_model.update_weights_prox(model=copy.deepcopy(global_model), global_round=epoch, mu = args.prox_mu, device=device)   
                if args.alg == 'Moon':
                    if len(Local_Weights)== 0:
                        w, loss = local_model.update_weights_CNN(model=copy.deepcopy(global_model), global_round=epoch, device=device)
                    else:
                        w, loss = local_model.update_weights_moon(
                            model=copy.deepcopy(global_model),previous_model_weight=copy.deepcopy(Local_Weights[epoch-1][idx]),temperature = args.moon_temp , global_round=epoch, mu = args.moon_mu, device=device)

                if args.alg == 'FedVAE':
                    w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch, u=args.vae_mu,device=device)

                
                if args.alg == 'FedMix':
                    w, loss = local_model.update_weights_fedmix(model=copy.deepcopy(global_model), global_round=epoch, images_means=images_means, labels_means=labels_means,lam = args.fedmix_lam, device=device)
                    
                    
                    
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

            # update global weights
            global_weights = average_weights(local_weights)

            # update global weights
            global_model.load_state_dict(global_weights)

            accuracy = 0
            cnt = 0
            global_model.eval()
            for cnt, (X,y) in enumerate(loader_test):
                X = X.to(device)
                y = y.double().to(device)
                if args.alg == 'FedAvg' or args.alg == 'Moon' or args.alg == 'FedProx' or args.alg == 'FedMix':
                    p = global_model(X)
                else:
                    m = torch.nn.Sigmoid()
                    X = m(X)
                    mu,logVar,p,_X = global_model(X)

                y_pred = p.argmax(1).double()
                accuracy += Accuracy(y,y_pred)
                cnt += 1
            print("accuracy of test at this round:",accuracy/cnt)
            ACC.append(accuracy/cnt)


    max_acc = max(ACC)
    end_time = time.time()
    print('max accuracy:', max_acc)
    print('running time: ', end_time - start_time)
    metrics['max_accuracy'] = max_acc
    metrics['time'] = end_time - start_time
    torch.save(global_model.state_dict(),checkpoint_dir+args_hash+'_model.pth')  
    with open(checkpoint_dir+args_hash+'_metrics.pkl', 'wb') as fp:
        pickle.dump(metrics, fp)
        
else:
    metrics = pickle.load(open(checkpoint_dir+args_hash+'_metrics.pkl', 'rb'))
    print(metrics)
