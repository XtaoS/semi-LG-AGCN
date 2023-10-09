import os
import random
import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.neighbors import kneighbors_graph
import torchvision.transforms as T
import torchvision.models as models
import scipy.io as sio
import scipy.sparse as sp
import argparse
from loadData import *

from models.query_models import VAE, Discriminator, GCN, GCNs, AGCN_FUN_M
from data.sampler import SubsetSequentialSampler

#init_seed = 123
#os.environ['PYTHONHASHSEED'] =str(init_seed)
#torch.manual_seed(init_seed)
#torch.cuda.manual_seed(init_seed)
#np.random.seed(init_seed)
#torch.cuda.manual_seed_all(init_seed)
#torch.backends.cudnn.deterministic =True

parser = argparse.ArgumentParser()
parser.add_argument("-l","--lambda_loss",type=float, default=1.1,
                    help="Adjustment graph loss parameter between the labeled and unlabeled")
parser.add_argument("-s","--s_margin", type=float, default=0.1,
                    help="Confidence margin of graph")
parser.add_argument("-n","--hidden_units", type=int, default=128,
                    help="Number of hidden units of the graph")
parser.add_argument("-r","--dropout_rate", type=float, default=0.3,
                    help="Dropout rate of the graph neural network")
parser.add_argument("-d","--dataset", type=str, default="IP",
                    help="")
parser.add_argument("-e","--no_of_epochs", type=int, default=20,
                    help="Number of epochs for the active learner")
parser.add_argument("-c","--cycles", type=int, default=5,
                    help="Number of active learning cycles")

args = parser.parse_args()

if __name__=="__main__":

    data_name = args.dataset
    num_classes = 16
    AddENDUM=100
    LR_GCN=0.001
    SUBSET=1500
    WDECAY=0.001
    BATCH=64
    CYCLE=8
    CUDA_VISIBLE_DEVICES = 0

    img_gyh = data_name+'_gyh'
    img_gt = data_name+'_gt'
    img_label=data_name+'_label'

    Data = load_HSI_data(data_name)
    [m, n]=np.shape(Data[img_gyh])
    matrix = [6, 22, 16, 7, 8, 14, 6, 9, 6, 18, 37, 11, 6, 21, 7, 6]
    # matrix = [24, 26, 16, 24, 24, 8, 24, 24, 24, 24, 24, 24, 10, 10, 14]
    label = Data[img_gt]
    indices=list(range(m))
    random.shuffle(indices)
    labeled_set=sampler_data(label, matrix,num_classes).astype('int64')
    #labeled_set = indices[:AddENDUM]
    #np.random.seed(123)
    random.shuffle(labeled_set)
    labeled_set=list(labeled_set)
    unlabeled_set = [x for x in indices if x not in labeled_set]
    intital_labeled = 300

    for cycle in range(CYCLE):
        #np.random.seed(123)
        random.shuffle(unlabeled_set)
        subset = unlabeled_set[:SUBSET]

        unlabeled_loader = Data[img_gyh][subset + labeled_set,:]
        unlabeled_loader= torch.from_numpy(unlabeled_loader).cuda()

        binary_labels = torch.cat((torch.zeros([SUBSET, 1]), (torch.ones([len(labeled_set), 1]))), 0)


        features = nn.functional.normalize(unlabeled_loader).to(torch.float32)
        features1= nn.functional.normalize(unlabeled_loader).to(torch.float32).cpu()
        adj_all= aff_to_adj(features)
        adj_nei= kneighbors_graph(features1, 10, mode='distance')
        adj_nei= adj_nei.A
        sigam=1
        for i in range(adj_nei.shape[0]):
            for j in range(adj_nei.shape[1]):
                if adj_nei[i][j]!=0:
                    adj_nei[i][j]=np.exp(-adj_nei[i][j]/(sigam*sigam))
        adj_d=np.sum(adj_nei,axis=1,keepdims=True)
        adj_d=np.diag(np.squeeze(adj_d**(-0.5)))
        adj_w=np.matmul(adj_nei,adj_d)
        adj_w=np.matmul(adj_d,adj_w)
        adj_nei=adj_w+np.eye(adj_w.shape[0])
        adj_nei=torch.from_numpy(adj_nei).cuda().to(torch.float32)

        gcn_module = AGCN_FUN_M(nfeat=features.shape[1],
                     nhid=args.hidden_units,
                     nclass=1,
                     dropout=args.dropout_rate).cuda()
        models = {'gcn_module': gcn_module}

        optim_backbone = optim.Adam(models['gcn_module'].parameters(), lr=LR_GCN, weight_decay=WDECAY)
        optimizers = {'gcn_module': optim_backbone}

        lbl = np.arange(SUBSET, SUBSET + cycle * AddENDUM + intital_labeled, 1)
        nlbl = np.arange(0, SUBSET, 1)

        for _ in range(1000):
            optimizers['gcn_module'].zero_grad()
            outputs, _, _ = models['gcn_module'](features, adj_all, adj_nei)
            lamda = args.lambda_loss
            loss = BCEAdjLoss(outputs, lbl, nlbl, lamda)
            loss.backward()
            optimizers['gcn_module'].step()

        models['gcn_module'].eval()
        with torch.no_grad():
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = features.cuda()
                labels = binary_labels.cuda()
        scores, _, feat = models['gcn_module'](inputs, adj_all, adj_nei)

        s_margin = args.s_margin
        scores_median = np.squeeze(
        torch.abs(scores[:SUBSET] - s_margin).detach().cpu().numpy())  # squeeze 删除维度为一的维度，abs求绝对值
        arg = np.argsort(-(scores_median))  # 进行排序，从小到大,加入负号使得从大到小

        print("Max confidence value: ", torch.max(scores.data))
        print("Mean confidence value: ", torch.mean(scores.data))
        preds = torch.round(scores)
        correct_labeled = (preds[SUBSET:, 0] == labels[SUBSET:, 0]).sum().item() / (cycle  * AddENDUM+intital_labeled)
        correct_unlabeled = (preds[:SUBSET, 0] == labels[:SUBSET, 0]).sum().item() / SUBSET
        correct = (preds[:, 0] == labels[:, 0]).sum().item() / (SUBSET + cycle  * AddENDUM + intital_labeled)
        print("Labeled classified: ", correct_labeled)
        print("Unlabeled classified: ", correct_unlabeled)
        print("Total classified: ", correct)

        labeled_set += list(torch.tensor(subset)[arg][-AddENDUM:].numpy())
        listd = list(torch.tensor(subset)[arg][:-AddENDUM].numpy())
        unlabeled_set = listd + unlabeled_set[SUBSET:]
        print(len(labeled_set), min(labeled_set), max(labeled_set))

    labeled_set2=Data[img_label][labeled_set,:]
    unlabeled_set2=Data[img_label][unlabeled_set,:]

    GCN_dataset=np.array(labeled_set2)
    GCN_dataset2=np.array(unlabeled_set2)
    sio.savemat("data/AGCN_TRdataset_M_900.mat", {'GCN_TRdataset': GCN_dataset})
    sio.savemat("data/AGCN_TEdataset_M_900.mat", {'GCN_TEdataset': GCN_dataset2})


