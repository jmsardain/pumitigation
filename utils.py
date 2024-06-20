import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from models import *
from torch_geometric.nn import aggr
import pickle
import torch.nn.functional as F
import math


class FocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


class LGK_LossFunction(nn.Module):
    def __init__(self, alpha=0.05, bandwith=0.1):
        super(LGK_LossFunction, self).__init__()
        self.alpha = alpha
        self.bandwith = bandwith
        self.pi = math.pi

    def forward(self, inputs, targets):
        targets = torch.clamp(targets, min=1e-6)  # Clamping to avoid division by zero
        inputs  = torch.clamp(inputs, min=1e-6)  # Clamping to avoid division by zero
        ratio = torch.clamp(torch.log(inputs / targets), min=1e-6)
        norm = -1 / (self.bandwith * torch.sqrt(torch.tensor(2 * self.pi)))
        gaussian_kernel = norm * torch.exp(-(ratio - 1) ** 2 / (2 * (self.bandwith ** 2)))
        leakiness = self.alpha * torch.abs(ratio - 1)
        lgk_loss = gaussian_kernel + leakiness
        loss = lgk_loss
        return loss.mean()


def get_latest_file(directory, DNNorRetrain=''):
    list_of_files = glob.glob(directory+'/'+DNNorRetrain+'_*.pt')
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def getPredictedResponse(num_features, device, dir_path, test_loader):

    predictions = []
    model = PUMitigation(num_features)
    model.to(device)

    ckpt_to_use = get_latest_file(dir_path, DNNorRetrain='Retrain')
    print(ckpt_to_use)
    checkpoint = torch.load(ckpt_to_use)
    model.load_state_dict(checkpoint['model_state_dict']) ## load model

    model.eval()
    for batch_x, batch_y in test_loader:
        out = model(batch_x).detach().cpu().numpy()
        predictions.append(out)

    predictions = np.concatenate(predictions)
    out = np.concatenate(predictions, axis=0)
    return out

def revertValue(x, mean, sigma):
    return torch.exp(sigma*x + mean)

with open('dict_mean_and_std.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

def loss_for_jetE(y_pred, y_true):
    # y_pred = torch.squeeze(y_pred)
    # y_true = torch.squeeze(y_true)
    y_true = torch.clamp(y_true, min=1e-6)  # Clamping to avoid division by zero
    y_pred = torch.clamp(y_pred, min=1e-6)  # Clamping to avoid division by zero
    ratio = torch.clamp(torch.log(y_pred / y_true), min=1e-6)
    loss = ratio
    return loss.mean()

def train_new(loader, model, device, optimizer):
    model.train()
    loss_all = 0
    loss_MSE = torch.nn.MSELoss()
    for idx, data in enumerate(loader):
        # if idx > 0: break
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        labels = torch.tensor(data.y, dtype=torch.float).to(device)
        labels = torch.reshape(labels, (int(list(labels.shape)[0]),1))
        ww = torch.tensor(data.weights, dtype=torch.float).to(device)
        ww = torch.reshape(ww, (int(list(labels.shape)[0]),1))

        loss = torch.nn.functional.binary_cross_entropy(out, labels, weight = ww)
        loss.backward()
        optimizer.step()
        loss_all += loss.item()
    return loss_all

def train_original(loader, model, device, optimizer):
    model.train()
    loss_all = 0
    # loss_func = FocalLoss(alpha=0.2, gamma=4)
    loss_func = LGK_LossFunction(alpha=0.05, bandwith=0.1)
    for idx, data in enumerate(loader):

        data = data.to(device)
        optimizer.zero_grad()
        out1, out2 = model(data.x, data.edge_index, data.JetRawPt)

        labels = torch.tensor(data.y, dtype=torch.float).to(device)
        labels = torch.reshape(labels, (int(list(labels.shape)[0]),1))
        ww = torch.tensor(data.weights, dtype=torch.float).to(device)
        ww = torch.reshape(ww, (int(list(labels.shape)[0]),1))

        Ejet_target = []
        Ejet_train  = []
        for i in range(len(data)):
            current_out = out2[data.ptr[i]:data.ptr[i + 1]]
            current_label = data.y[data.ptr[i]:data.ptr[i + 1]]
            current_clusterE = data.ClusterEOriginal[data.ptr[i]:data.ptr[i + 1]]
            Ejet_target.append(torch.sum( current_clusterE * current_label       ))
            Ejet_train.append(torch.sum(  current_clusterE * current_out.view(-1)))

        Ejet_train  = torch.stack(Ejet_train)
        Ejet_target = torch.stack(Ejet_target)

        loss1 = torch.nn.BCELoss(weight=ww)(out1, labels)
        loss2 = loss_func(Ejet_train, Ejet_target)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        loss_all += loss.item()
    return loss_all


def validate_original(loader, model, device ):
    model.eval()
    loss_all = 0
    # loss_func = FocalLoss(alpha=0.2, gamma=4)
    loss_func = LGK_LossFunction(alpha=0.05, bandwith=0.1)
    for data in loader:
        data = data.to(device)
        out1, out2 = model(data.x, data.edge_index, data.JetRawPt)
        out1 = out1.view(-1, out1.shape[-1])
        out2 = out2.view(-1, out2.shape[-1])

        labels = torch.tensor(data.y, dtype=torch.float).to(device)
        labels = torch.reshape(labels, (int(list(labels.shape)[0]),1))
        ww = torch.tensor(data.weights, dtype=torch.float).to(device)
        ww = torch.reshape(ww, (int(list(labels.shape)[0]),1))

        Ejet_target = []
        Ejet_train  = []
        for i in range(len(data)):
            current_out = out2[data.ptr[i]:data.ptr[i + 1]]
            current_label = data.y[data.ptr[i]:data.ptr[i + 1]]
            current_clusterE = data.ClusterEOriginal[data.ptr[i]:data.ptr[i + 1]]
            Ejet_target.append(torch.sum( current_clusterE * current_label       ))
            Ejet_train.append(torch.sum(  current_clusterE * current_out.view(-1)))

        Ejet_train  = torch.stack(Ejet_train)
        Ejet_target = torch.stack(Ejet_target)

        # loss1 = loss_func(out, labels)
        # loss = loss1 * data.pt_weight.mean()
        # loss = torch.nn.NLLLoss(weight = torch.tensor([0.00001, 1.0])(out, labels))
        loss1 = torch.nn.BCELoss(weight=ww)(out1, labels)
        # loss2 = 5*loss_for_jetE(Ejet_train, Ejet_target)
        loss2 = loss_func(Ejet_train, Ejet_target)
        loss = loss1 + loss2
        loss_all += loss.item()
    return loss_all


def validate_new(loader, model, device):
    model.eval()
    loss_all = 0
    loss_MSE = torch.nn.MSELoss()
    for idx, data in enumerate(loader):
        # if idx > 0: break
        data = data.to(device)
        out = model(data.x, data.edge_index)
        labels = torch.tensor(data.y, dtype=torch.float).to(device)
        ## reshape now
        out = out.view(-1, out.shape[-1])
        labels = torch.reshape(labels, (int(list(labels.shape)[0]),1))
        ww = torch.tensor(data.weights, dtype=torch.float).to(device)
        ww = torch.reshape(ww, (int(list(labels.shape)[0]),1))
        loss = torch.nn.functional.binary_cross_entropy(out, labels, weight = ww)
        loss_all += loss.item()
    return loss_all




def train(loader, model, device, optimizer):
    model.train()
    loss_all = 0
    clf, reg = 0, 0
    for idx, data in enumerate(loader):
        # if idx > 0: break
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        clusterEDNN = data.ClusterEDNN
        labels = torch.tensor(data.y, dtype=torch.float).to(device)
        Ejet_target = []
        Ejet_train  = []
        ## regression part
        for i in range(len(data)):
            # if i > 0: break;
            current_out = out[data.ptr[i]:data.ptr[i + 1]]
            current_clusterEDNN = clusterEDNN[data.ptr[i]:data.ptr[i + 1]]
            mask = labels[data.ptr[i]:data.ptr[i + 1]] < 0.1
            Ejet_target.append(torch.sum(current_clusterEDNN[mask]))
            Ejet_train.append(torch.sum(current_clusterEDNN * (1 - current_out.view(-1))))

        Ejet_train  = torch.stack(Ejet_train)
        Ejet_target = torch.stack(Ejet_target)
        # print(Ejet_target)
        # reshape now
        labels = torch.reshape(labels, (int(list(labels.shape)[0]),1))
        ww = torch.tensor(data.weights, dtype=torch.float).to(device)
        ww = torch.reshape(ww, (int(list(labels.shape)[0]),1))
        #
        loss_reg = loss_for_jetE(Ejet_train, Ejet_target)
        # loss_clf = torch.nn.functional.binary_cross_entropy(out, labels, weight = ww)
        loss_clf = torch.nn.BCELoss(weight=ww)(out, labels)
        loss = loss_clf - 0.1 * loss_reg
        loss.backward()
        optimizer.step()
        loss_all += loss.item()
    # print("loss_clf: {}     loss_reg: {}".format(clf, reg))
    return loss_all

def validate(loader, model, device):
    model.eval()
    loss_all = 0
    loss_MSE = torch.nn.MSELoss()
    for idx, data in enumerate(loader):
        # if idx > 0: break
        data = data.to(device)
        out = model(data.x, data.edge_index)
        clusterEDNN = data.ClusterEDNN
        labels = torch.tensor(data.y, dtype=torch.float).to(device)
        Ejet_target = []
        Ejet_train  = []
        ## regression part
        for i in range(len(data)):
            current_out = out[data.ptr[i]:data.ptr[i + 1]]
            current_clusterEDNN = clusterEDNN[data.ptr[i]:data.ptr[i + 1]]
            mask = labels[data.ptr[i]:data.ptr[i + 1]] < 0.1
            Ejet_target.append(torch.sum(current_clusterEDNN[mask]))
            Ejet_train.append(torch.sum(current_clusterEDNN * (1 - current_out.view(-1))))

        Ejet_train = torch.stack(Ejet_train)
        Ejet_target =torch.stack(Ejet_target)

        ## reshape now
        out = out.view(-1, out.shape[-1])
        labels = torch.reshape(labels, (int(list(labels.shape)[0]),1))
        ww = torch.tensor(data.weights, dtype=torch.float).to(device)
        ww = torch.reshape(ww, (int(list(labels.shape)[0]),1))

        loss_reg = loss_for_jetE(Ejet_train, Ejet_target)
        # loss = F.binary_cross_entropy(output, new_y, weight = new_w)
        # loss_clf = torch.nn.functional.binary_cross_entropy(out, labels, weight = ww)
        loss_clf = torch.nn.BCELoss(weight=ww)(out, labels)
        # loss = torch.nn.functional.binary_cross_entropy(out, labels)
        loss = loss_clf - 0.1* loss_reg
        # loss = loss_clf + l2reg
        loss_all += loss.item()
    return loss_all


# def train_old(loader, model, device, optimizer):
#     model.train()
#     loss_all = 0
#     from torch_geometric.nn import aggr
#     sum1 = aggr.SumAggregation()
#     loss_MSE = torch.nn.MSELoss()
#
#     for idx, data in enumerate(loader):
#         if idx > 0: break
#         data = data.to(device)
#         optimizer.zero_grad()
#         out = model(data.x, data.edge_index)
#         print(len(data))
#
#         labels = torch.tensor(data.y, dtype=torch.float).to(device)
#         jetCnt = data.jetCnt
#         jetCnt = torch.tensor(jetCnt, dtype=torch.int64)
#         clusterE = data.x[:,0]
#         clusterE = revertValue(clusterE, loaded_dict["clusterE"][0], loaded_dict["clusterE"][1]).to(device)
#         clusterEDNN  = clusterE / data.REPredicted
#         clusterEDNN = torch.reshape(clusterEDNN, (int(list(clusterEDNN.shape)[0]),1))
#
#         Ejet_target  = sum1(clusterEDNN[labels < 0.1], jetCnt[labels < 0.1] )
#         Ejet_train   = sum1(clusterEDNN*(1-out), jetCnt)
#
#         ## reshape now
#         labels = torch.reshape(labels, (int(list(labels.shape)[0]),1))
#         jetCnt = torch.reshape(jetCnt, (int(list(jetCnt.shape)[0]),1))
#         ww = torch.tensor(data.weights, dtype=torch.float).to(device)
#         ww = torch.reshape(ww, (int(list(labels.shape)[0]),1))
#
#         loss_reg = loss_MSE(Ejet_train, Ejet_target)
#         # loss = F.binary_cross_entropy(output, new_y, weight = new_w)
#         loss_clf = torch.nn.functional.binary_cross_entropy(out, labels, weight = ww)
#         # loss = torch.nn.functional.binary_cross_entropy(out, labels)
#         loss = loss_clf + loss_reg
#         loss.backward()
#         optimizer.step()
#         loss_all += loss.item()
#     return loss_all
#
#

# def train(loader, model, device, optimizer):
#     model.train()
#     loss_all = 0
#     sum1 = aggr.SumAggregation()
#     loss_MSE = torch.nn.MSELoss()
#
#     with open('dict_mean_and_std.pkl', 'rb') as f:
#         loaded_dict = pickle.load(f)
#     for data in loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         out = model(data.x, data.edge_index)
#         labels = torch.tensor(data.y, dtype=torch.float).to(device)
#         labels = torch.reshape(labels, (int(list(labels.shape)[0]),1))
#
#         # jetCnt = data.jetCnt
#         # jetCnt = torch.tensor(jetCnt, dtype=torch.int64)
#         # jetCnt = torch.reshape(jetCnt, (int(list(jetCnt.shape)[0]),1))
#
#
#         ## get Ejet train
#         clusterE = data.x[:,0]
#         clusterE = revertValue(clusterE, loaded_dict["clusterE"][0], loaded_dict["clusterE"][1]).to(device)
#         clusterE = clusterE*out
#         Ejet_train = torch.tensor([clusterE[graph['nodes']].sum() for graph in data])
#
#
#         ## get Ejet target
#         clusterE_target = clusterE[labels_tensor < 0.1]
#         clusterE_target = revertValue(clusterE_target, loaded_dict["clusterE"][0], loaded_dict["clusterE"][1]).to(device)
#         Ejet_target = torch.tensor([clusterE_target[graph['nodes']].sum() for graph in data])
#
#
#
#
#         ## reshape now
#         ww = torch.tensor(data.weights, dtype=torch.float).to(device)
#         ww = torch.reshape(ww, (int(list(labels.shape)[0]),1))
#
#         ## define losses
#         loss_reg = loss_MSE(Ejet_train, Ejet_target)
#         loss_clf = torch.nn.functional.binary_cross_entropy(out, labels, weight = ww)
#         loss = loss_clf + loss_reg
#         loss.backward()
#         optimizer.step()
#         loss_all += loss.item()
#     return loss_all


# def validate_old(loader, model, device, optimizer):
#     model.eval()
#     loss_all = 0
#     sum1 = aggr.SumAggregation()
#     loss_MSE = torch.nn.MSELoss()
#     for data in loader:
#         data = data.to(device)
#         out = model(data.x, data.edge_index)
#
#         labels = torch.tensor(data.y, dtype=torch.float).to(device)
#         jetCnt = data.jetCnt
#         jetCnt = torch.tensor(jetCnt, dtype=torch.int64)
#
#         clusterE = data.x[:,0]
#         clusterE = revertValue(clusterE, loaded_dict["clusterE"][0], loaded_dict["clusterE"][1]).to(device)
#         clusterEDNN  = clusterE / data.REPredicted
#
#         clusterEDNN = torch.reshape(clusterEDNN, (int(list(clusterEDNN.shape)[0]),1))
#
#         Ejet_target  = sum1(clusterEDNN[labels < 0.1], jetCnt[labels < 0.1] )
#         Ejet_train   = sum1(clusterEDNN*(1-out), jetCnt)
#         # print(Ejet_train)
#         # print(Ejet_target)
#
#         # print(Ejet_train.shape)
#         # print(Ejet_target.shape)
#         out = out.view(-1, out.shape[-1])
#         labels = torch.reshape(labels, (int(list(labels.shape)[0]),1))
#         ww = torch.tensor(data.weights, dtype=torch.float).to(device)
#         ww = torch.reshape(ww, (int(list(labels.shape)[0]),1))
#
#         loss_clf = torch.nn.functional.binary_cross_entropy(out, labels, weight = ww)
#         loss_reg = loss_MSE(Ejet_train, Ejet_target)
#         loss = loss_clf + loss_reg
#         # loss = torch.nn.functional.binary_cross_entropy(out, labels)
#
#         # ce_loss = torch.nn.functional.binary_cross_entropy(out, labels)
#         # p_t = torch.exp(-ce_loss)
#         # alpha = 0.25
#         # gamma = 0.5
#         # loss = (1 - p_t) ** gamma
#         # loss = - alpha * loss
#         # loss = torch.mean(loss * ce_loss)
#
#         loss_all += loss.item()
#     return loss_all


def plot_ROC_curve(loader, model, device, edges_or_nodes, outdir=''):
    model.eval()
    nodes_out = torch.tensor([])
    labels_test = torch.tensor([])
    Cluster_E = torch.tensor([])
    Cluster_Eta = torch.tensor([])
    loss_all = 0
    for data in loader:
        data = data.to(device)

        if edges_or_nodes=="edges":
            out = model(data.x, data.edge_index, data.edge_attr)
        else:
            out = model(data.x, data.edge_index, data.JetRawPt)
        out = out.view(-1, out.shape[-1])

        labels = torch.tensor(data.y, dtype=torch.float).to(device)
        labels = torch.reshape(labels, (int(list(labels.shape)[0]),1))
        ww = torch.tensor(data.weights, dtype=torch.float).to(device)
        ww = torch.reshape(ww, (int(list(labels.shape)[0]),1))

        Cluster_E_temp = torch.tensor(data.x[:,0], dtype=torch.float).cpu()#to(device)
        Cluster_E_temp = torch.reshape(Cluster_E_temp, (int(list(labels.shape)[0]),1))
        Cluster_E = torch.cat((Cluster_E.clone().detach(), Cluster_E_temp.clone().detach()), 0)
        ClusterEta_temp = torch.tensor(data.x[:,1], dtype=torch.float).cpu()#.to(device)
        ClusterEta_temp = torch.reshape(ClusterEta_temp, (int(list(labels.shape)[0]),1))
        Cluster_Eta = torch.cat((Cluster_Eta.clone().detach(), ClusterEta_temp.clone().detach()), 0)

        nodes_out = torch.cat((nodes_out.clone().detach(), out.clone().detach().cpu()), 0)
        labels_test = torch.cat((labels_test.clone().detach(), labels.clone().detach().cpu()), 0)

        # loss = torch.nn.functional.binary_cross_entropy(out, labels, weight = ww)
        # loss = torch.nn.functional.binary_cross_entropy(out, labels)
        ce_loss = torch.nn.functional.binary_cross_entropy(out, labels)
        p_t = torch.exp(-ce_loss)
        alpha = 0.25
        gamma = 0.5
        loss = (1 - p_t) ** gamma
        loss = - alpha * loss
        loss = torch.mean(loss * ce_loss)
        loss_all += loss.item()

    nodes_out = torch.squeeze(nodes_out)
    labels_test = torch.squeeze(labels_test)
    df_out = pd.DataFrame()
    df_out['labels'] = labels_test
    df_out['score'] = nodes_out
    df_out['ClusterE'] = Cluster_E
    df_out['ClusterEta'] = Cluster_Eta

    fpr, tpr, thresholds = roc_curve(labels_test, nodes_out )
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    roc_auc = roc_auc_score(labels_test, nodes_out )
    ax.annotate(f'Area Under Curve {roc_auc:.4f}' , xy=(310, 120), xycoords='axes points',
                size=12, ha='right', va='top',
                bbox=dict(boxstyle='round', fc='w'))
    plt.xlabel("False positive rate", size=14)
    plt.ylabel("True positive rate", size=14)
    plt.savefig(outdir+'_roc_curve.png')
    plt.show()

def train_edge(loader, model, device, optimizer):
    model.train()
    loss_all = 0
    for idx, data in enumerate(loader):
        #print(len(data))
        data = data.to(device)
        optimizer.zero_grad()
        #out = model(data.x, data.edge_index)
        out = model(data.x, data.edge_index, data.edge_attr)
        #out = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device))

        labels = torch.tensor(data.y, dtype=torch.float).to(device)
        labels = torch.reshape(labels, (int(list(labels.shape)[0]),1))
        ww = torch.tensor(data.weights, dtype=torch.float).to(device)
        ww = torch.reshape(ww, (int(list(labels.shape)[0]),1))


        # loss = F.binary_cross_entropy(output, new_y, weight = new_w)
        loss = torch.nn.BCELoss(weight=ww)(out, labels)
        loss.backward()
        optimizer.step()
        loss_all += loss.item()
    return loss_all

def validate_edge(loader, model, device):
    model.eval()
    loss_all = 0
    for idx, data in enumerate(loader):
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)
        labels = torch.tensor(data.y, dtype=torch.float).to(device)
        ## reshape now
        out = out.view(-1, out.shape[-1])
        labels = torch.reshape(labels, (int(list(labels.shape)[0]),1))
        ww = torch.tensor(data.weights, dtype=torch.float).to(device)
        ww = torch.reshape(ww, (int(list(labels.shape)[0]),1))

        # loss = F.binary_cross_entropy(output, new_y, weight = new_w)
        # loss_clf = torch.nn.functional.binary_cross_entropy(out, labels, weight = ww)
        loss_clf = torch.nn.BCELoss(weight=ww)(out, labels)
        # loss = torch.nn.functional.binary_cross_entropy(out, labels)
        loss = loss_clf
        # loss = loss_clf + l2reg
        loss_all += loss.item()
    return loss_all

def train_edge_2(epoch, loader, model, device, optimizer, optimizer2, optimizer3):
    model.train()
    loss_all = 0
    for data in loader:
        #print(len(data))
        if len(data)< 512 : continue
        #data = data.to(device)
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        #out = model(data.x, data.edge_index)
        #out = model(data.x, data.edge_index, data.edge_attr)
        out = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device))

        labels = torch.tensor(data.y, dtype=torch.float).to(device)
        labels = torch.reshape(labels, (int(list(labels.shape)[0]),1))
        ww = torch.tensor(data.weights, dtype=torch.float).to(device)
        ww = torch.reshape(ww, (int(list(labels.shape)[0]),1))


        # loss = F.binary_cross_entropy(output, new_y, weight = new_w)
        loss = torch.nn.functional.binary_cross_entropy(out, labels, weight = ww)
        loss.backward()
        if epoch < 5:
            optimizer3.step()
        elif epoch >= 5 and epoch > 10:
            optimizer2.step()
        elif epoch >= 10:
            optimizer.step()
        loss_all += loss.item()
    return loss_all



def validate_edge_old(loader, model, device, optimizer):
    model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        #out = model(data.x, data.edge_index)
        out = model(data.x, data.edge_index, data.edge_attr)
        out = out.view(-1, out.shape[-1])

        labels = torch.tensor(data.y, dtype=torch.float).to(device)
        labels = torch.reshape(labels, (int(list(labels.shape)[0]),1))
        ww = torch.tensor(data.weights, dtype=torch.float).to(device)
        ww = torch.reshape(ww, (int(list(labels.shape)[0]),1))

        loss = torch.nn.functional.binary_cross_entropy(out, labels, weight = ww)
        loss_all += loss.item()
    return loss_all
