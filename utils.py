
import numpy as np
import torch
import scipy
from torch.utils.data import Dataset
from torchvision.utils import save_image
from torch.optim import Optimizer
from keras.utils import to_categorical
from sklearn.cluster import KMeans
from collections import Counter
    
def target_distribution(q):
    """
    compute the target distribution of t distribtion
    """
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


def acc(y_true,y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.type(torch.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.shape[0]):
        w[y_pred[i], y_true[i]] += 1
    ind = scipy.optimize.linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.shape[0]

def Precision(y,y_predict):
    leng = len(y)
    nomaly = sum(y,1).item()
    miss = 0
    for i in range(leng):
        if y[i] == 1 and y_predict[i] == 0:
            miss +=1
    return (nomaly-miss)/nomaly

def Accuracy(y,y_predict):
    leng = len(y)
    miss = 0
    for i in range(leng):
        if not y[i]==y_predict[i]:
            miss +=1
    return (leng-miss)/leng

def getLocalCenter(model,trainloader,major_classes,device):
        data_len = len(trainloader)
        class0 = []
        class1 = []
        class2 = []
        for cnt, (X,y) in enumerate(trainloader):
            X = X.to(device)
            code = model.module.Encoder(X).to(device)
            for i in range(len(code)):
                if y[i] == major_classes[0]:
                    class0.append(code[i].detach().cpu().numpy())
                if y[i] == major_classes[1]:
                    class1.append(code[i].detach().cpu().numpy())
                if y[i] == major_classes[2]:
                    class2.append(code[i].detach().cpu().numpy())
                    
        class0 = torch.DoubleTensor(class0)
        center0 = class0.mean(axis=0)
        std0 = class0.std(axis=0)
        class1 = torch.DoubleTensor(class1)
        center1 = class1.mean(axis=0)
        std1 = class1.std(axis=0)
        class2 = torch.DoubleTensor(class2)
        center2 = class2.mean(axis=0)
        std2 = class2.std(axis=0)
        del y
        del X
        return [center0,center1,center2],[std0,std1,std2]

def getLocalCenters(model,Loaders_train,num_users,Major_classes,device):
    centers = []
    stds = []
    for idx in range(num_users):
        center,std = getLocalCenter(model,Loaders_train[idx],Major_classes[idx],device)
        centers.append(center) 
        stds.append(std)    
    return centers,stds

def allocatePairs(num_users):
    num_pairs = int(num_users/2)
    Pairs, all_clients = {}, [i for i in range(num_users)]
    for i in range(num_pairs):
        client_pair = np.random.choice(all_clients, 2,replace=False)
        client1 = client_pair[0]
        client2 = client_pair[1]
        Pairs[int(client1)] = int(client2)
        Pairs[int(client2)] = int(client1)
        pair = set(client_pair)
        all_clients = list(set(all_clients) - pair)
    return Pairs 


    
def save_decoded_image(args,img, name):
    if args.dataset == 'CIFAR100' or args.dataset == 'CIFAR10':
        img = img.view(img.size(0), 3, 32, 32)
    if args.dataset == 'FMNIST':
        img = img.view(img.size(0), 1, 28, 28)
    save_image(img, name)


def reparameterize(mu, logVar):
    #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
    std = torch.exp(logVar/2)
    eps = torch.randn_like(std)
    return mu + std * eps

def getLocalMean(model,trainloader,major_classes,device):
        data_len = len(trainloader)
        Mu = []
        Var = []
        for i in range(len(major_classes)):
            mu = []
            var = []
            Mu.append(mu)
            Var.append(var)
        for cnt, (X,y) in enumerate(trainloader):
            X = X.to(device)
            m = torch.nn.Sigmoid()
            X = m(X)
            mu,logVar,p,_ = model(X)
            var = logVar
            for i in range(len(mu)):
                for j in range(len(major_classes)):
                    if y[i] == major_classes[j]:
                        Mu[j].append(mu[i].detach().cpu().numpy())
                        Var[j].append(var[i].detach().cpu().numpy())
                        break
        
        for i in range(len(major_classes)):
            Mu[i] = torch.DoubleTensor(Mu[i])     
            Var[i] = torch.DoubleTensor(Var[i])  
    
        del y
        del X
        return Mu,Var
    
def getLocalMeans(Models,Loaders_train,num_users,Major_classes,device):
    Mus = []
    Vars = []
    for idx in range(num_users):
        Mu, Var = getLocalMean(Models[idx],Loaders_train[idx],Major_classes[idx],device)
        Mus.append(Mu) 
        Vars.append(Var)
        
    return Mus, Vars

def getLocalMeans_global(model,Loaders_train,num_users,Major_classes,device):
    Mus = []
    Vars = []
    for idx in range(num_users):
        Mu, Var = getLocalMean(model,Loaders_train[idx],Major_classes[idx],device)
        Mus.append(Mu) 
        Vars.append(Var)
        
    return Mus, Vars

class ConcatDataset(Dataset):
    def __init__(self, dataloader1, dataloader2,device):
        
        for idx,(X, y) in enumerate(dataloader1):
            m = torch.nn.Sigmoid()
            X = m(X)
            if idx == 0: 
                X0 = X.to(device)
                y0 = y.to(device)
            else:
                X0 = torch.cat((X0,X.to(device)),dim = 0)
                y0 = torch.cat((y0,y.to(device)),dim = 0)
                
        for idx,(X, y) in enumerate(dataloader2):
            X0 = torch.cat((X0,X.to(device)),dim = 0)
            y0 = torch.cat((y0,y.to(device)),dim = 0)
            
        self.X = X0
        self.y = y0


    def __getitem__(self, idx):
        return self.X[idx],self.y[idx]

    def __len__(self):
        return len(self.y)
    
class GenData(Dataset):
    def __init__(self,
                 args,
                 model_generate,
                 model_recognize,
                 Mean,
                 Var,
                 major_classes, 
                 num_gen,
                 std,
                 device,
                 idx):
        super(GenData, self).__init__()
        self.Mean = Mean
        self.Var = Var
        self.num_gen = num_gen
        X = []
        y = []
        Generate_Latent = []
        print(major_classes)
        m = len(major_classes)
        for i in range(m):
            count = 0
            z = []
            for j in range(len(Mean[i])):
                lantency = reparameterize(Mean[i][j],Var[i][j])
                z.append(lantency)
            if len(z) == 0:
                continue
            index_range = range(len(z))
            if len(z) > 20:
                sample_idex = np.random.choice(index_range,size=int(0.1*len(z)))
            else:
                sample_idex = np.random.choice(index_range,size=len(z))
            z = [z[x] for x in sample_idex]    
            z = torch.stack(z).to(device)  
            mean = torch.mean(z)
            var = torch.var(z,dim=0)
            counts = 0
            iteration = 0
            while counts < num_gen and iteration < 50:
                iteration += 1
                z_noise = std*torch.randn(num_gen,args.code_len).to(device)   
                z_mean =  mean.repeat(num_gen,1)
                z = z_mean + z_noise

                X = model_generate.module.Decoder(z)
                mu,logVar,p,_ = model_recognize(X)
                p = np.argmax(p.cpu().detach().numpy(),1)
                for j in range(num_gen):
                    if p[j] == major_classes[i]:
                        counts += 1
                        Generate_Latent.append(z[j])
                        y.append(major_classes[i])

        
        Generate_Latent = torch.stack(Generate_Latent).to(device)
        X = model_generate.module.Decoder(Generate_Latent).detach()
        save_decoded_image(args,X.cpu().data, name='./Generated_'+ args.dataset + '/X{}.png'.format(idx))
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):      
        return self.X[idx],self.y[idx]
    