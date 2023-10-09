clear all;
close all;clc;
addpath(genpath('./data'))

load AGCN_TEdataset.mat;
load AGCN_TRdataset.mat;
load IP_gyh.mat;
load IP_gt.mat;
I = IP_gyh;
[m, n, z] = size(I);
gt= IP_gt;
GCN_TRdataset=double(GCN_TRdataset).'+1;
GCN_TEdataset=double(GCN_TEdataset).'+1;

gt=reshape(gt.',m*n,1);

TR=zeros(m,n);
TE=zeros(m,n);
TR(GCN_TRdataset)=gt(GCN_TRdataset);
TE(GCN_TEdataset)=gt(GCN_TEdataset);

I2d = hyperConvert2d(I);
for i = 1 : z
    I2d(i,:) = mat2gray(I2d(i,:));%¹éÒ»»¯²Ù×÷
end
TR2d = hyperConvert2d(TR);
TE2d = hyperConvert2d(TE);
I = hyperConvert3d(I2d, m, n, z);
[X_train, X_test, X_train_P, X_test_P, Y_train, Y_test] = TR_TE_Generation2d_CNN(I, TR, TE, 3);

 save('.\data/X_train.mat','X_train');
 save('.\data/X_test.mat','X_test');
 save('.\data/Y_train.mat','Y_train');
 save('.\data/Y_test.mat','Y_test');