#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse


# In[1]:


def args_parser():
    parser = argparse.ArgumentParser()

    #Model specific parameters
    parser.add_argument('--code_len', default=128,
                        help='length of code ')
    #Data specifc paremeters
    parser.add_argument('--dataset', default='ELECTRO',
                        help='ELECTRO, RSNA') 
    #Training specifc parameters
    parser.add_argument('--log_frq', type=int, default=5,
                        help='frequency of logging')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')  
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='gadient clipping')        
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_sh_rate', type=int, default=50,
                        help='number of steps to drop the lr')
    parser.add_argument('--use_lrschd', action="store_true", default=False,
                        help='Use lr rate scheduler')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='drop out rate for training') 
    parser.add_argument('--classify_loss_weight', type=float, default=0.5,
                        help='weight for output of clustering layer') 
    parser.add_argument('--clusters', default=2,
                        help='number of clusters from data') 
    parser.add_argument('--tag', default='centralized',
                        help='centralized, federated')
    parser.add_argument('--num_users', default=32,
                        help='number of local models')
    parser.add_argument('--update_frac', default=1,
                        help='frac of local models to update')
    parser.add_argument('--local_ep', default=5,
                        help='iterations of local updating')
    parser.add_argument('--eval_only',action="store_true", default=False,
                        help='load the trained model and see the metric')

    args = parser.parse_args('')
    return args

