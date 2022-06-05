import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable

INPUT_DIM = 224
MAX_PIXEL_VAL = 255

class Dataset(data.Dataset):
    def __init__(self, datadirs,use_gpu):
        super().__init__()
        self.use_gpu = use_gpu

        label_dict = {}
        self.paths = []

        for i, line in enumerate(open('TrainingData.csv').readlines()):
            if i == 0:
                continue
            line = line.strip().split(',')
            path = line[8]
            label = line[5]
            label_dict[path] = int(label)

        for dir in datadirs:
            for file in os.listdir(dir):
                self.paths.append(dir+'/'+file)
                
        self.labels = [label_dict[path[17:]] for path in self.paths]

        neg_weight = np.mean(self.labels)
        self.weights = [neg_weight, 1 - neg_weight]

    def weighted_loss(self, prediction, target):
        weights_npy = np.array([self.weights[int(t[0])] for t in target.data])
        weights_tensor = torch.FloatTensor(weights_npy)
        if self.use_gpu:
            weights_tensor = weights_tensor.cuda()
        loss = F.binary_cross_entropy_with_logits(prediction, target, weight=Variable(weights_tensor))
        return loss

    def __getitem__(self, index):
        path = self.paths[index]
        with open(path,'rb') as file_handler:
            vol=pickle.load(file_handler).astype(np.int32)
            if(vol.shape[0]>31):
                diff = vol.shape[0]-30
                vol = vol[int(diff/2):-int(diff/2),:,:]
        
        #fname = path[20:-4]

        # crop middle
        pad = int((vol.shape[2] - INPUT_DIM)/2)
        vol = vol[:,pad:-pad,pad:-pad]
        
        # standardize
        vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * MAX_PIXEL_VAL
        vol = (vol - np.mean(vol)) / np.std(vol)
        vol = np.stack((vol,)*3, axis=1)

        vol_tensor = torch.FloatTensor(vol)
        label_tensor = torch.FloatTensor([self.labels[index]])
        
        return vol_tensor, label_tensor

    def __len__(self):
        return len(self.paths)

def load_data(use_gpu=False):
    train_dirs    = ['vol02/axial_left','vol02/coron_left','vol02/sagit_left',
                     'vol03/axial_left','vol03/coron_left','vol03/sagit_left',
                     'vol04/axial_left','vol04/coron_left','vol04/sagit_left',
                     'vol05/axial_left','vol05/coron_left','vol05/sagit_left']
    
    valid_dirs    = ['vol01/axial_left']
    
    train_dataset = Dataset(train_dirs, use_gpu)
    valid_dataset = Dataset(valid_dirs, use_gpu)

    train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=8, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=1, num_workers=8, shuffle=False)
   
    return train_loader, valid_loader