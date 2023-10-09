import random

import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
from sklearn.neighbors import kneighbors_graph
import math

def Con2Numpy(var_name):
    path = './/data//'
    dataFile = path + var_name
    data = scio.loadmat(dataFile)
    x = data[var_name]
    x1 = x.astype(float)
    return x1

def Con1Numpy(var_name):
    path='.//data//cnn_data//'
    dataFile = path + var_name
    data = scio.loadmat(dataFile)
    x = data[var_name]
    x1 = x.astype(float)
    return x1

def load_HSI_data( data_name ):
    Data = dict()
    img_gyh = data_name+'_gyh'
    img_gt = data_name+'_gt'
    img_label= data_name+'_label'
    Data[img_gt] = np.array(Con2Numpy(img_gt), dtype='int')
    Data[img_gyh] = Con2Numpy(img_gyh)
    [r, c, l]=np.shape(Data[img_gyh])
    [r, c] = np.shape(Data[img_gt])
    Data2 = np.reshape(Data[img_gt], [r * c])
    Data1=np.reshape(Data[img_gyh], [r*c, l])
    reigon1 = np.argwhere(Data2>0)
    Data3=Data1[np.squeeze(reigon1),:]
    Data4=Data2[reigon1]
    Data[img_gyh]=Data3
    Data[img_gt]=Data4
    Data[img_label]=reigon1
    return Data

def load_GCN_data(method):
    Data=dict()
    if method=='Dataset_N':
        img_gyh = 'ALL_X'
        img_gt = 'ALL_Y'
    elif method=='Dataset_A':
        img_gyh = 'ALL_X_A'
        img_gt = 'ALL_Y_A'
    elif method == 'Dataset_M':
        img_gyh = 'ALL_X_M'
        img_gt = 'ALL_Y_M'
        #img_gt = 'label_reconstruct'
       # img_Y = 'Y_predict'
    elif method == 'Dataset_C':
        img_gyh = 'ALL_X_C'
        img_gt = 'ALL_Y_C'
    Data[img_gt]=np.array(Con2Numpy(img_gt),dtype='int')
    Data[img_gyh]=Con2Numpy(img_gyh)
    #Data[img_Y]=np.array(Con2Numpy(img_Y),dtype='int')
    return Data

def load_Fun_data(img_gyh,img_gt,img_cnn):
    Data=dict()
    Data[img_gt]=np.array(Con2Numpy(img_gt),dtype='int')
    Data[img_gyh]=Con2Numpy(img_gyh)
    Data[img_cnn]=Con2Numpy(img_cnn)
    return Data

def load_CNN_data(random):
    Data=dict()
    if random==1:
        img_gyh='X_CNN_random1'
        img_gt='Y_CNN_random1'
    elif random==2:
        img_gyh = 'X_CNN_random2'
        img_gt = 'Y_CNN_random2'
    elif random == 3:
        img_gyh = 'X_CNN_random3'
        img_gt = 'Y_CNN_random3'
    elif random==4:
        img_gyh = 'X_CNN_random4'
        img_gt = 'Y_CNN_random4'
    elif random==5:
        img_gyh = 'X_CNN_random5'
        img_gt = 'Y_CNN_random5'
    elif random==6:
        img_gyh = 'X_CNN_random6'
        img_gt = 'Y_CNN_random6'
    elif random==7:
        img_gyh = 'X_CNN_random7'
        img_gt = 'Y_CNN_random7'
    elif random==8:
        img_gyh = 'X_CNN_random8'
        img_gt = 'Y_CNN_random8'
    elif random==9:
        img_gyh = 'X_CNN_random9'
        img_gt = 'Y_CNN_random9'
    elif random==10:
        img_gyh = 'X_CNN_random10'
        img_gt = 'Y_CNN_random10'
    elif random == 11:
        img_gyh = 'X_CNN_random11'
        img_gt = 'Y_CNN_random11'
    elif random == 12:
        img_gyh = 'X_CNN_random12'
        img_gt = 'Y_CNN_random12'
    elif random == 13:
        img_gyh = 'X_CNN_M'
        img_gt = 'ALL_Y_M'
    Data[img_gt]=np.array(Con1Numpy(img_gt),dtype='int')
    Data[img_gyh] = Con1Numpy(img_gyh)
    return Data

def aff_to_adj(x, y=None):
    x = x.detach().cpu().numpy()
    adj = np.matmul(x, x.transpose())
    adj +=  -1.0*np.eye(adj.shape[0])
    adj_diag = np.sum(adj, axis=0) #rowise sum
    adj = np.matmul(adj, np.diag(1/adj_diag))
    adj = adj + np.eye(adj.shape[0])
    adj = torch.Tensor(adj).cuda()
    return adj


def BCEAdjLoss(scores, lbl, nlbl, l_adj):
    lnl = torch.log(scores[lbl])
    lnu = torch.log(1 - scores[nlbl])
    labeled_score = torch.mean(lnl)
    unlabeled_score = torch.mean(lnu)
    bce_adj_loss = -labeled_score - l_adj*unlabeled_score
    return bce_adj_loss

def convert_to_one_hot(Y, C):
    [m, n]=np.shape(Y)
    p=np.zeros([m,C])
    for i in range(m):
        p[i,Y[i,n-1]-1]=1
    Y=p
    return Y

def aff_to_knei(traininput):
    features1 = nn.functional.normalize(traininput).to(torch.float32).cpu()
    adj_nei = kneighbors_graph(features1, 10, mode='distance')
    adj_nei = adj_nei.A
    sigam = 1
    for i in range(adj_nei.shape[0]):
        for j in range(adj_nei.shape[1]):
            if adj_nei[i][j] != 0:
                adj_nei[i][j] = np.exp(-adj_nei[i][j] / (sigam * sigam))
    adj_d = np.sum(adj_nei, axis=1, keepdims=True)
    adj_d = np.diag(np.squeeze(adj_d ** (-0.5)))
    adj_w = np.matmul(adj_nei, adj_d)
    adj_w = np.matmul(adj_d, adj_w)
    adj_nei = adj_w + np.eye(adj_w.shape[0])
    adj_nei = torch.from_numpy(adj_nei).cuda().to(torch.float32)
    return adj_nei

def sampler_data(input, matrix, numclass):
    #input is the label of original data
    y3=[]
    for i in range(1, numclass+1):
        y1=np.argwhere(input==i)[:,0]
        random.shuffle(y1)
        y2=y1[:matrix[i-1]]
        y3=np.append(y3,y2)
    sampler=y3
    return sampler

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def mini_batch_GCN(input, label, L, minibatches):

    m = input.shape[0]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    inputs = input[permutation, :]
    shuffle_label = label[permutation, :].reshape((m,label.shape[1]))
    shuffle_L1=L[permutation,:].reshape((L.shape[0],L.shape[1]))
    shuffle_L=shuffle_L1[:,permutation].reshape((L.shape[0], L.shape[1]))

    num_complete_minibatches = math.floor(m / minibatches)
    for num in range(0, num_complete_minibatches):
        minibatch_x = inputs[num * minibatches: num * minibatches + minibatches, :]
        minibatch_y = shuffle_label[num * minibatches: num * minibatches + minibatches, :]
        minibatch_l = shuffle_L[num * minibatches: num * minibatches + minibatches, num * minibatches: num * minibatches + minibatches]
        mini_batch = (minibatch_x, minibatch_y, minibatch_l)
        mini_batches.append(mini_batch)
    mini_batch=(input,label,L)
    mini_batches.append(mini_batch)
    return mini_batches
