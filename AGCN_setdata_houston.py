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
import torchvision.transforms as T
import torchvision.models as models
import scipy.io as sio
import argparse
from loadData import *

from models.query_models import VAE, Discriminator, GCN, GCNs
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
parser.add_argument("-d","--dataset", type=str, default="HU",
                    help="")
parser.add_argument("-e","--no_of_epochs", type=int, default=20,
                    help="Number of epochs for the active learner")
parser.add_argument("-c","--cycles", type=int, default=5,
                    help="Number of active learning cycles")

args = parser.parse_args()

def Con3Numpy(var_name,data_name):
    path = './/data//'
    dataFile = path + var_name
    data = scio.loadmat(dataFile)
    x = data[data_name]
    x1 = x.astype(float)
    x1=np.array(x1, dtype='int')
    return x1

if __name__=="__main__":

    data_name = args.dataset
    num_classes = 15
    AddENDUM = 150
    LR_GCN=0.001
    SUBSET=1500
    WDECAY=0.001
    BATCH = 64
    CYCLE = 8
    CUDA_VISIBLE_DEVICES = 0

    img_gyh = data_name+'_gyh'
    img_gt = data_name+'_gt'
    img_label=data_name+'_label'

    Data = load_HSI_data(data_name)
    [m, n]=np.shape(Data[img_gyh])
    matrix = [24, 26, 16, 24, 24, 8, 24, 24, 24, 24, 24, 24, 10, 10, 14]
    # matrix = [6, 22, 16, 7, 8, 14, 6, 9, 6, 18, 37, 11, 6, 21, 7, 6]
    label = Data[img_gt]
    indices = list(range(m))
    # np.random.seed(123)
    random.shuffle(indices)


    labeled_set = sampler_data(label, matrix, num_classes).astype('int64')
    # labeled_set = indices[:AddENDUM]
    # np.random.seed(123)
    random.shuffle(labeled_set)
    labeled_set = list(labeled_set)
    unlabeled_set = [x for x in indices if x not in labeled_set]


    for cycle in range(CYCLE):
        #np.random.seed(123)
        random.shuffle(unlabeled_set)
        subset = unlabeled_set[:SUBSET]

        unlabeled_loader = Data[img_gyh][subset + labeled_set,:]
        unlabeled_loader= torch.from_numpy(unlabeled_loader).cuda()

        binary_labels = torch.cat((torch.zeros([SUBSET, 1]), (torch.ones([len(labeled_set), 1]))), 0)


        features = nn.functional.normalize(unlabeled_loader).to(torch.float32)
        adj=aff_to_adj(features)

        gcn_module = GCN(nfeat=features.shape[1],
                     nhid=args.hidden_units,
                     nclass=1,
                     dropout=args.dropout_rate).cuda()
        models = {'gcn_module': gcn_module}

        optim_backbone = optim.Adam(models['gcn_module'].parameters(), lr=LR_GCN, weight_decay=WDECAY)
        optimizers = {'gcn_module': optim_backbone}

        lbl = np.arange(SUBSET, SUBSET + (cycle+1) * AddENDUM, 1)
        nlbl = np.arange(0, SUBSET, 1)

        for _ in range(400):
            optimizers['gcn_module'].zero_grad()
            outputs, _, _ = models['gcn_module'](features, adj)
            lamda = args.lambda_loss
            loss = BCEAdjLoss(outputs, lbl, nlbl, lamda)
            loss.backward()
            optimizers['gcn_module'].step()

        models['gcn_module'].eval()
        with torch.no_grad():
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = features.cuda()
                labels = binary_labels.cuda()
        scores, _, feat = models['gcn_module'](inputs, adj)

        s_margin = args.s_margin
        scores_median = np.squeeze(
        torch.abs(scores[:SUBSET] - s_margin).detach().cpu().numpy())  # squeeze 删除维度为一的维度，abs求绝对值
        arg = np.argsort(-(scores_median))  # 进行排序，从小到大,加入负号使得从大到小

        print("Max confidence value: ", torch.max(scores.data))
        print("Mean confidence value: ", torch.mean(scores.data))
        preds = torch.round(scores)
        correct_labeled = (preds[SUBSET:, 0] == labels[SUBSET:, 0]).sum().item() / ((cycle + 1) * AddENDUM)
        correct_unlabeled = (preds[:SUBSET, 0] == labels[:SUBSET, 0]).sum().item() / SUBSET
        correct = (preds[:, 0] == labels[:, 0]).sum().item() / (SUBSET + (cycle + 1) * AddENDUM)
        print("Labeled classified: ", correct_labeled)
        print("Unlabeled classified: ", correct_unlabeled)
        print("Total classified: ", correct)

        labeled_set += list(torch.tensor(subset)[arg][-AddENDUM:].numpy())
        listd = list(torch.tensor(subset)[arg][:-AddENDUM].numpy())
        unlabeled_set = listd + unlabeled_set[SUBSET:]
        print(len(labeled_set), min(labeled_set), max(labeled_set))

    labeled_set2 = Data[img_label][labeled_set,:]
    unlabeled_set2 = Data[img_label][unlabeled_set,:]

    GCN_dataset = np.array(labeled_set2)
    GCN_dataset2 = np.array(unlabeled_set2)
    sio.savemat("data/AGCN_TRdataset_hou200.mat", {'GCN_TRdataset': GCN_dataset})
    sio.savemat("data/AGCN_TEdataset_hou200.mat", {'GCN_TEdataset': GCN_dataset2})



