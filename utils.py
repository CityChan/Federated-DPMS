
import numpy as np
import torch
import scipy
from torch.utils.data import Dataset
from torchvision.utils import save_image


    
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

class GenData(Dataset):
    def __init__(self, 
                 model,
                 centers,
                 stds,
                 major_classes, 
                 num_gen,
                 device):
        super(GenData, self).__init__()
        self.model = model.to(device)
        self.centers = centers
        self.stds = stds
        self.num_gen = num_gen
        num_each_class = int(self.num_gen/3)
        codes = []
        y = []
        for i in range(num_each_class):
            code = torch.normal(centers[0], np.absolute(stds[0]))
            codes.append(code)
            y.append(major_classes[0])
            code = torch.normal(centers[1], np.absolute(stds[1]))
            codes.append(code)
            y.append(major_classes[1])
            code = torch.normal(centers[2], np.absolute(stds[2]))
            codes.append(code)
            y.append(major_classes[2])
        codes = torch.stack(codes).to(device)
        y = torch.DoubleTensor(y).to(device)
        X = model.module.Decoder(codes).detach()
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):      
        return self.X[idx],self.y[idx]

class GenCode(Dataset):
    def __init__(self, 
                 model,
                 centers,
                 stds,
                 major_classes, 
                 num_gen,
                 device):
        super(GenCode, self).__init__()
        self.model = model.to(device)
        self.centers = centers
        self.stds = stds
        self.num_gen = num_gen
        num_each_class = int(self.num_gen/3)
        codes = []
        y = []
        for i in range(num_each_class):
            code = torch.normal(centers[0], np.absolute(stds[0]))
            codes.append(code)
            y.append(major_classes[0])
            code = torch.normal(centers[1], np.absolute(stds[1]))
            codes.append(code)
            y.append(major_classes[1])
            code = torch.normal(centers[2], np.absolute(stds[2]))
            codes.append(code)
            y.append(major_classes[2])
        codes = torch.stack(codes).to(device)
        y = torch.DoubleTensor(y).to(device)
        self.codes = codes
        self.y = y
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):      
        return self.codes[idx],self.y[idx]
    
def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 32, 32)
    save_image(img, name)