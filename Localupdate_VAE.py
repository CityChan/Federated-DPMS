
import numpy as np
import torch
import scipy
from torch.utils.data import Dataset
import torch
import copy
import torch.nn as nn
from sklearn.cluster import KMeans
import torch.optim as optim
import torch.nn.functional as F
from utils import acc,target_distribution, Accuracy

class LocalUpdate(object):
    """
    This class is for train the local model with input global model(copied) and output the updated weight
    args: argument 
    Loader_train,Loader_val,Loaders_test: input for training and inference
    user: the index of local model
    idxs: the index for data of this local model
    logger: log the loss and the process
    """
    def __init__(self, args, Loader_train, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader = Loader_train
        self.idxs = idxs
        self.bce = nn.BCELoss(size_average=False) 
        self.ce = nn.CrossEntropyLoss() 

        
    def update_weights(self, model, global_round,u,device):
        model.cuda()
        model.train()
        epoch_loss = []
        optimizer = optim.Adam(model.parameters(),lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(device)
                y = y.long().to(device)
                m = torch.nn.Sigmoid()
                X = m(X) 
                optimizer.zero_grad()
                mu,logVar,p,_X = model(X)
                y_pred = p.argmax(1)
                kl_divergence = -0.5 * (1 + logVar - mu.pow(2) - logVar.exp()).mean()
                bce =  self.bce(_X,X)/ X.size(0)
                ce = self.ce(p,y)
                loss = u*(bce + kl_divergence) + ce
                loss.backward()
                optimizer.step()
                accuracy = Accuracy(y,y_pred)
                if batch_idx % 10 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]KLD: {:.6f} BCE: {:.6f} Acc: {:.6f}'.format(
                        global_round, iter, batch_idx * len(X),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), kl_divergence,bce.item(),accuracy))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
   
    
    
    def update_weights_gen(self, model, loader_gen,u, global_round,device):
        model.cuda()
        model.train()
        epoch_loss = []
        optimizer = optim.Adam(model.parameters(),lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(device)
                y = y.long().to(device)
                m = torch.nn.Sigmoid()
                X = m(X) 
                optimizer.zero_grad()
                mu,logVar,p,_X = model(X)
                y_pred = p.argmax(1)
                kl_divergence = -0.5 * (1 + logVar - mu.pow(2) - logVar.exp()).mean()
                bce =  self.bce(_X,X)/ X.size(0)
                ce = self.ce(p,y)
                loss = u*(bce + kl_divergence) + ce
                loss.backward()
                optimizer.step()
                accuracy = Accuracy(y,y_pred)
                if batch_idx % 10 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]KLD: {:.6f} BCE: {:.6f} Acc: {:.6f}'.format(
                        global_round, iter, batch_idx * len(X),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), kl_divergence,bce.item(),accuracy))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

            for batch_idx, (X, y) in enumerate(loader_gen):
                X = X.to(device)
                y = y.long().to(device)
                m = torch.nn.Sigmoid()
                X = m(X) 
                optimizer.zero_grad()
                mu,logVar,p,_X = model(X)
                y_pred = p.argmax(1)
                kl_divergence = -0.5 * (1 + logVar - mu.pow(2) - logVar.exp()).mean()
                bce =  self.bce(_X,X)/ X.size(0)
                ce = self.ce(p,y)
                loss = u*(bce + kl_divergence) + ce
                loss.backward()
                optimizer.step()
                accuracy = Accuracy(y,y_pred)
                if batch_idx % 10 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]KLD: {:.6f} BCE: {:.6f} Acc: {:.6f}'.format(
                        global_round, iter, batch_idx * len(X),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), kl_divergence,bce.item(),accuracy))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))



        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    
    def update_weights_norm(self, model, global_round,u,device):
        model.cuda()
        model.train()
        epoch_loss = []
        optimizer = optim.Adam(model.parameters(),lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(device)
                y = y.long().to(device)
                optimizer.zero_grad()
                mu,logVar,p,_X = model(X)
                y_pred = p.argmax(1)
                kl_divergence = -0.5 * (1 + logVar - mu.pow(2) - logVar.exp()).mean()
                bce =  self.bce(_X,X)/ X.size(0)
                ce = self.ce(p,y)
                loss = u*(bce + kl_divergence) + ce
                loss.backward()
                optimizer.step()
                accuracy = Accuracy(y,y_pred)
                if batch_idx % 10 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]KLD: {:.6f} BCE: {:.6f} Acc: {:.6f}'.format(
                        global_round, iter, batch_idx * len(X),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), kl_divergence,bce.item(),accuracy))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def update_weights_contrastive(self, model, Mus, Vars,global_round,u,device):
        model.cuda()
        model.train()
        epoch_loss = []
        optimizer = optim.Adam(model.parameters(),lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(device)
                y = y.long().to(device)
                m = torch.nn.Sigmoid()
                X = m(X) 
                optimizer.zero_grad()
                mu,logVar,p,_X = model(X)
                y_pred = p.argmax(1)
                kl_divergence = -0.5 * (1 + logVar - mu.pow(2) - logVar.exp()).mean()
                bce =  self.bce(_X,X)/ X.size(0)
                ce = self.ce(p,y)
                loss = u*(bce + kl_divergence) + ce
                loss.backward()
                optimizer.step()
                accuracy = Accuracy(y,y_pred)
                if batch_idx % 10 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]KLD: {:.6f} BCE: {:.6f} Acc: {:.6f}'.format(
                        global_round, iter, batch_idx * len(X),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), kl_divergence,bce.item(),accuracy))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    
   