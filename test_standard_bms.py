import collections
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
import torch.optim as optim
from numpy import linalg as LA
import argparse

use_gpu = torch.cuda.is_available()

def parse_args():
    parser = argparse.ArgumentParser(description= 'few-shot script')
    parser.add_argument('--dataset', default='mini', help='model: mini/tiered/cub/cifar-fs')
    parser.add_argument('--model', default='wrn', help='model: wrn') 
    parser.add_argument('--shot', default=1, type=int, help='1/5')
    parser.add_argument('--run', default=10000, type=int, help='600/1000/10000')
    parser.add_argument('--way', default=5, type=int)
    parser.add_argument('--query', default=15, type=int)
    parser.add_argument('--method', default='BMS', help='BMS/BMS_')
    parser.add_argument('--preprocess', default='PEME')
    parser.add_argument('--step', default=20, type=int)
    parser.add_argument('--mmt', default=0.8, type=float)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--lam', default=8.5, type=float)
    parser.add_argument('--epoch', default=0, type=int)
    return parser.parse_args()

class DataSet:
    data: None
    labels: None
        
    def __init__(self, data=None, n_shot=1, n_ways=5, n_queries=15):
        self.data = data
        self.n_shot = n_shot
        self.n_ways = n_ways
        if self.data is not None:
            self.n_runs = data.size(0)
            self.n_samples = data.size(1)
            self.n_feat = data.size(2)
            self.n_lsamples = n_ways*n_shot
            self.n_queries = n_queries
            self.n_usamples = n_ways*n_queries
            self.generateLabels()
            if self.n_samples != self.n_lsamples + self.n_usamples:
                print("Invalid settings: queries incorrect wrt size")
                self.exit()
                
    def cuda(self):
        self.data = self.data.cuda()
        self.labels = self.labels.cuda()
    
    def cpu(self):
        self.data = self.data.cpu()
        self.labels = self.labels.cpu()        
    
    def generateLabels(self):
        self.labels = torch.arange(self.n_ways)\
            .view(1,1,self.n_ways)\
            .expand(self.n_runs,self.n_shot+self.n_queries,self.n_ways)\
            .clone().view(self.n_runs, self.n_samples)
    def printState(self):
        print("DataSet: {}-shot, {}-ways, {}-queries, {}-runs, {}-feats".format( \
             self.n_shot, self.n_ways, self.n_queries, self.n_runs, self.n_feat))
        print("\t {}-labelled {}-unlabelled {}-tot".format( \
              self.n_lsamples, self.n_usamples, self.n_samples))

class BaseModel:
    def __init__(self, ds):
        self.ds = ds
        
    # SHOULD not be override! 
    #   this should be done through getScores() overriding
    def getProbas(self, scoresRescale=1, forceLabelledToOne=True):
        scores = self.getScores()
        p_xj = F.softmax(-scores*scoresRescale, dim=2)
        if forceLabelledToOne:
            p_xj[:,:self.ds.n_lsamples].fill_(0)
            p_xj[:,:self.ds.n_lsamples].scatter_(2,self.ds.labels[:,:self.ds.n_lsamples].unsqueeze(2), 1)
            
        return p_xj

class TrainedLinRegModel(BaseModel):
    def __init__(self, ds, useBias=False):
        super(TrainedLinRegModel, self).__init__(ds)
        self.mus = None         # shape [n_runs][n_ways][n_nfeat]
        self.ds = ds
        self.weights = torch.Tensor(ds.n_runs, ds.n_feat, ds.n_ways)
        self.scalers = torch.Tensor(ds.n_runs)
        
    def cuda(self):
        self.mus = self.mus.cuda()
        self.weights = self.weights.cuda()
        self.scalers = self.scalers.cuda()
        
    def cpu(self):
        self.mus = self.mus.cpu()
        self.weights = self.weights.cpu()
        self.scalers = self.scalers.cpu()
        
    # intitialise params of linReg from pre-computed mus
    def initParams(self):
        self.scalers.fill_(1)
        self.mus = self.mus/self.mus.norm(dim=2, keepdim=True) 
        cdata = self.mus.permute(0,2,1)
        self.weights.copy_(cdata)
        
    def initFromLabelledDatas(self):
        ds = self.ds
        self.mus = ds.data\
            .reshape(ds.n_runs, ds.n_shot+ds.n_queries,ds.n_ways, ds.n_feat)[:,:ds.n_shot,]\
            .mean(1)

        self.initParams()

    def getParameters(self, asNNParameter=False):
        params = [self.weights, self.scalers]
        if asNNParameter:
            pp = [nn.Parameter(t.clone()) for t in params]
            params = pp
        
        return params            
    
    def getScores(self):
        params = self.getParameters()
        ds = self.ds
        scores = 1 - ds.data.matmul(self.weights)
        
        return scores
    
    def train(self, wsamples, trainCfg, wmasker=None, updateWeights=True):
        ds = self.ds

        # computing emus
        emus = wsamples.permute(0,2,1).matmul(ds.data).div(wsamples.sum(dim=1).unsqueeze(2))
        emus = emus/emus.norm(dim=2, keepdim=True)   
        cdata = emus.permute(0,2,1) #[10000, 640, 5]
        
        mparameters = self.getParameters(asNNParameter=True)
                
        # get initialisation from centroids estimate
        mparameters[0].data.copy_(cdata)        
        mparameters[1].data.fill_(1)
        optimizer = torch.optim.SGD(mparameters, lr=trainCfg['lr'], momentum=trainCfg['mmt'])
        
        for epoch in range(trainCfg['epochs']):
            optimizer.zero_grad()
            scores = ds.data.matmul(mparameters[0])
            scores = scores / mparameters[0].norm(dim=1, keepdim=True)
            scores = scores * mparameters[1].unsqueeze(1).unsqueeze(1)

            output = F.log_softmax(scores, dim=2)
            loss_train = -output.mul(wsamples).sum(2).mean(1).sum(0)
            
            loss_train.backward()
            optimizer.step()
            mparameters[0].data.div_(mparameters[0].data.norm(dim=1, keepdim=True))
            
        if updateWeights:
            self.weights.copy_(mparameters[0].data)
            
            
# =========================================
#    Optimization routines
# =========================================

class Optimizer:
    def __init__(self, ds, wmasker=None):
        self.ds = ds
        self.wmasker = wmasker
        if self.wmasker is None:
            self.wmasker = SimpleWMask()
        
    def getAccuracy(self, probas):
        olabels = probas.argmax(dim=2)
        matches = self.ds.labels.eq(olabels).float()
        acc_test = matches[:,self.ds.n_lsamples:].mean(1)  

        m = acc_test.mean().item()
        pm = acc_test.std().item() *1.96 / math.sqrt(acc_test.size(0))
        return m, pm
    
# =========================================
#    Class to define samples mask for Centroid computations
# =========================================

class SimpleWMask:
    """ class that selects which samples to be used for centroid computatoin 
    Default implementation use probas as wmask
    """
    def __init__(self, ds):
        self.ds = ds
        self.doSinkhorn = True
        self.nIter = 50
    
    def BMS_(self, p, nIter=None): 
        global epoch
        ds = self.ds
        target = ds.n_queries
        if nIter is None: 
            nIter = self.nIter
        op = p[:,ds.n_lsamples:]
        
        for iter in range(nIter):
            wp = op.div(op.sum(1, keepdim=True)/target)
            op = wp.div(wp.sum(2, keepdim=True))
    
        wm = p.clone()
        wm[:,ds.n_lsamples:] = op
        wm[:,:ds.n_lsamples] = 0
        wm[:,:ds.n_lsamples].scatter_(2,ds.labels[:,:ds.n_lsamples].unsqueeze(2), 1)
        return wm
    
    def BMS(self, p, minSize, nIter=None):
        global epoch
        ds = self.ds
        wm_total = []
        n_runs, n, m = p.shape 
        
        if nIter is None: 
            nIter = self.nIter
        op = p[:,ds.n_lsamples:]
        
        for iter in range(nIter):
            op = op.div(op.sum(2, keepdim=True))
            mask = (op.sum(1, keepdim=True) < minSize.unsqueeze(1).unsqueeze(1)).all(dim=1).int() # [10000, 5]
            mask_inv = (op.sum(1, keepdim=True) >= minSize.unsqueeze(1).unsqueeze(1)).all(dim=1).int()
            temp = op * mask.unsqueeze(1)
            temp = temp.div((temp+1e-20).sum(1, keepdim=True)/minSize.unsqueeze(1).unsqueeze(1))
            op = op * mask_inv.unsqueeze(1) + temp * mask.unsqueeze(1)
                
        wm = p.clone()
        wm[:,ds.n_lsamples:] = op
        wm[:,:ds.n_lsamples] = 0
        wm[:,:ds.n_lsamples].scatter_(2, ds.labels[:,:ds.n_lsamples].unsqueeze(2), 1)
        return wm
    
    def fastOT(self, p, minSize):
        if params.method == 'BMS':
            q = self.BMS(p, minSize)
        elif params.method == 'BMS_':
            q = self.BMS_(p)
        return q
    
    def getMinsize(self, p):
        mask = p.new_zeros(p.shape)
        mask.scatter_(2, p.argmax(dim=2, keepdim=True), 1)
        nsamplesInCluster = mask.sum(1)
        minSize = nsamplesInCluster.min(1)[0] 
        
        return minSize
    
    def getWMask(self, probas, minSize, epochInfo=None):
        
        if self.doSinkhorn:
            return self.fastOT(probas, minSize)
        else:
            return probas


# ========================================
#      loading datas

def reloadRuns(shot=1, n_ways=5, n_queries=15, n_runs=10000):
    (datas, labels) = torch.load("cache/runs{}_s{}_w{}_q{}_r{}".format(n_runs, shot, n_ways, n_queries, n_runs))
    print("-- loaded datas and labels size:")
    print(datas.size())
    print(labels.size())
    
    return datas, labels

def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def centerDatas(datas):
    
    datas = datas - datas.mean(1, keepdim=True)
    return datas

def scaleEachUnitaryDatas(datas):
    
    norms = datas.norm(dim=2, keepdim=True)
    return datas/norms

def QRreduction(datas):
    
    #ndatas = torch.linalg.qr(datas.permute(0,2,1), mode='reduced').R
    ndatas = torch.qr(datas.permute(0,2,1)).R
    ndatas = ndatas.permute(0,2,1)
    return ndatas

def getRunSet(n_shot, n_ways, n_queries, n_runs, preprocess='PEME', dataset='mini', model='wrn'):
    import FSLTask
    cfg = {'shot':n_shot, 'ways':n_ways, 'queries':n_queries, 'runs':n_runs}
    load = dataset + '_' + model
    FSLTask.loadDataSet(load)
    ndatas = FSLTask.GenerateRunSet(cfg=cfg)
    ds = DataSet(ndatas.permute(0,2,1,3).reshape(n_runs, -1, ndatas.size(3)), \
                 n_shot=n_shot, n_ways=n_ways, n_queries=n_queries)
    
    if preprocess == 'R':
        print("--- preprocess: QR decomposition")
        ds.data = QRreduction(ds.data)
        ds.n_feat = ds.data.size(2)
        return ds
    
    if 'P' not in preprocess:
        print("--- preprocess: QR decomposition")
        ds.data = QRreduction(ds.data)
        ds.n_feat = ds.data.size(2)

    for p in preprocess:
        if p=='P':
            print("--- preprocess: Power transform")
            ds.data = torch.sqrt(ds.data+1e-6)
            
        elif p=="M":
            print("--- preprocess: Mean subtraction")
            ds.data = centerDatas(ds.data)
            print("--- preprocess: QR decomposition")
            ds.data = QRreduction(ds.data)
            ds.n_feat = ds.data.size(2)
        elif p=="E":
            print("--- preprocess: Euclidean normalization")
            ds.data = scaleEachUnitaryDatas(ds.data)
        else:
            print("unknown preprocessing!!")
            torch.exit()
            
    return ds
                    
if __name__ == '__main__':
# ---- data loading
    params = parse_args()
    n_shots = params.shot
    n_runs = params.run
    n_ways = params.way
    n_queries = params.query
    dataset = getRunSet(n_shot=n_shots, n_ways=n_ways, n_queries=n_queries, n_runs=n_runs, preprocess=params.preprocess, dataset=params.dataset, model=params.model)
    dataset.printState()
    dataset.cuda()
    
    wmasker = SimpleWMask(dataset)
    optim = Optimizer(dataset, wmasker=wmasker)
    mm = TrainedLinRegModel(dataset)
    mm.initFromLabelledDatas()
    mm.cuda()
    
    probas = mm.getProbas()
    init_acc = optim.getAccuracy(probas)
    print("initialisation model accuracy", init_acc)
    
    minSize = (torch.ones(n_runs) * n_shots).cuda()
    for iter in range(params.step):
        probas = mm.getProbas(scoresRescale=params.lam)
        probas = wmasker.getWMask(probas, minSize)
        minSize = wmasker.getMinsize(probas)

        trainCfg = {'lr':params.lr, 'mmt':params.mmt, 'epochs':params.epoch}
        mm.train(probas, trainCfg, wmasker=wmasker, updateWeights=True)
        sink_acc = optim.getAccuracy(probas)
        print(iter, sink_acc)

