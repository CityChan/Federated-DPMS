#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse


# In[1]:


def args_parser():
    parser = argparse.ArgumentParser()

    #Data specifc paremeters
    parser.add_argument('--dataset', default='CIFAR10',
                        help='CIFAR10, CIFAR100, FMNIST') 
    #Training specifc parameters
    parser.add_argument('--log_frq', type=int, default=5,
                        help='frequency of logging')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')  
#     parser.add_argument('--clip_grad', type=float, default=None,
#                         help='gadient clipping')        
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_sh_rate', type=int, default=10,
                        help='number of steps to drop the lr')
#     parser.add_argument('--use_lrschd', action="store_true", default=False,
#                         help='Use lr rate scheduler')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='drop out rate for training') 
    parser.add_argument('--tag', default='centralized',
                        help='centralized, federated')
    parser.add_argument('--num_users', default=10,
                        help='number of local models')
    parser.add_argument('--update_frac', default=1,
                        help='frac of local models to update')
    parser.add_argument('--local_ep', default=5,
                        help='iterations of local updating')
    parser.add_argument('--beta', default=0.5,
                        help='beta for non-iid distribution')
    parser.add_argument('--seed', default=0,
                        help='random seed for generating datasets')
    parser.add_argument('--mini', default=1,
                        help='sample rate of each local data')
    parser.add_argument('--moon_mu', default=5,
                        help='weight for moon term')
    parser.add_argument('--moon_temp', default=0.5,
                        help='temperture for moon')
    parser.add_argument('--prox_mu', default=0.001,
                        help='weight for prox term')
    parser.add_argument('--pretrain', default=20,
                        help='pretrain epochs for vae')
    parser.add_argument('--gen_num', default=50,
                        help='number of generating images')
    parser.add_argument('--std', default=4,
                        help='std for generating means')
    parser.add_argument('--inputsize', default=32,
                        help='input size')
    parser.add_argument('--inputchannel', default=3,
                        help='input channel')
    parser.add_argument('--classes', default=10,
                        help='number of classes')
    parser.add_argument('--code_len', default=32,
                        help='length of code')
    parser.add_argument('--alg', default='FedAvg',
                        help='FedAvg, FedProx, Moon, FedVAE, DPMS')
    parser.add_argument('--eval_only', action="store_true", default=False,help='evaluate the model')

    

    args = parser.parse_args('')
    return args

