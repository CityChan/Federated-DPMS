
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
#         self.mse = nn.MSELoss() 

        
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
#                 if batch_idx % 10 == 0:
#                     print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]KLD: {:.6f} BCE: {:.6f} Acc: {:.6f}'.format(
#                         global_round, iter, batch_idx * len(X),
#                         len(self.trainloader.dataset),
#                         100. * batch_idx / len(self.trainloader), kl_divergence,bce.item(),accuracy))
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
#                 if batch_idx % 10 == 0:
#                     print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]KLD: {:.6f} BCE: {:.6f} Acc: {:.6f}'.format(
#                         global_round, iter, batch_idx * len(X),
#                         len(self.trainloader.dataset),
#                         100. * batch_idx / len(self.trainloader), kl_divergence,bce.item(),accuracy))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
#     def update_weights_CNN(self, model, global_round,u,device):
#         model.cuda()
#         model.train()
#         epoch_loss = []
#         optimizer = optim.Adam(model.parameters(),lr=self.args.lr)
#         scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
#         for iter in range(self.args.local_ep):
#             batch_loss = []
#             for batch_idx, (X, y) in enumerate(self.trainloader):
#                 X = X.to(device)
#                 y = y.long().to(device)
#                 optimizer.zero_grad()
#                 p = model(X)
#                 y_pred = p.argmax(1)
#                 ce = self.ce(p,y)
#                 loss = ce
#                 loss.backward()
#                 optimizer.step()
#                 accuracy = Accuracy(y,y_pred)
# #                 if batch_idx % 10 == 0:
# #                     print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]Acc: {:.6f}'.format(
# #                         global_round, iter, batch_idx * len(X),
# #                         len(self.trainloader.dataset),
# #                         100. * batch_idx / len(self.trainloader),accuracy))
#                 self.logger.add_scalar('loss', loss.item())
#                 batch_loss.append(loss.item())
#             epoch_loss.append(sum(batch_loss)/len(batch_loss))

#         return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def update_weights_moon(self, model,previous_model_weight, temperature,global_round, mu,device):
        model.cuda()
        model.train()
        previous_model = copy.deepcopy(model)
        previous_model.load_state_dict(previous_model_weight)
        global_model = copy.deepcopy(model)
        cos=torch.nn.CosineSimilarity(dim=-1)
        global_model.eval()
        previous_model.eval()
        
        epoch_loss = []
        optimizer = optim.Adam(model.parameters(),lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(device)
                y = y.to(device).long()
                optimizer.zero_grad()
                p1 = model(X)
                p2= global_model(X)
                p3 = previous_model(X)
                p1 = p1.double()
                p2 = p2.double()
                p3 = p3.double()
                posi = cos(p1, p2)
                logits = posi.reshape(-1,1)
                nega = cos(p1, p3)
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)
                logits /= temperature
                labels = torch.zeros(X.size(0)).cuda().long()
                loss2 =  self.ce(logits, labels)
                
                y_pred = p1.argmax(1)
                loss1 = self.ce(p1,y)
                loss = loss1 + mu*loss2
                loss.backward()
                optimizer.step()
                accuracy = Accuracy(y,y_pred)

#                 if batch_idx % 10 == 0:
#                     print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t moon_loss: {:.6f} Acc: {:.6f}'.format(
#                         global_round, iter, batch_idx * len(X),
#                         len(self.trainloader.dataset),
#                         100. * batch_idx / len(self.trainloader), loss.item(),loss2.item(),accuracy))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def update_weights_prox(self, model,global_round, mu,device):
        model.cuda()
        model.train()
        global_model = copy.deepcopy(model)
        global_model.eval()
        global_weight_collector = list(global_model.parameters())
        epoch_loss = []
        optimizer = optim.Adam(model.parameters(),lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(device)
                y = y.to(device).long()
                optimizer.zero_grad()
                p = model(X).double()
                
                y_pred = p.argmax(1)
                loss1 = self.ce(p,y)
                fed_prox_reg = 0.0
                for param_index, param in enumerate(model.parameters()):
                    fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss = loss1 + mu*fed_prox_reg
                loss.backward()
                optimizer.step()

#                 if batch_idx % 10 == 0:
#                     print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t prox_loss: {:.6f}'.format(
#                         global_round, iter, batch_idx * len(X),
#                         len(self.trainloader.dataset),
#                         100. * batch_idx / len(self.trainloader), loss.item(),fed_prox_reg.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def update_weights_CNN(self, model, global_round, device):
        model.cuda()
        model.train()
        epoch_loss = []
        optimizer = optim.Adam(model.parameters(),lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(device)
                y = y.to(device).long()
                optimizer.zero_grad()
                p = model(X).double()
                y_pred = p.argmax(1)
                loss = self.ce(p,y)
                loss.backward()
                optimizer.step()

#                 if batch_idx % 10 == 0:
#                     print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                         global_round, iter, batch_idx * len(X),
#                         len(self.trainloader.dataset),
#                         100. * batch_idx / len(self.trainloader), loss.item()))
#                 self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    