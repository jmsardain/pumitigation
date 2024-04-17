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
# alpha = 0.25
# gamma = 0.5
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
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

        labels = torch.tensor(data.y, dtype=torch.float).to(device)
        labels = torch.reshape(labels, (int(list(labels.shape)[0]),1))
        ww = torch.tensor(data.weights, dtype=torch.float).to(device)
        ww = torch.reshape(ww, (int(list(labels.shape)[0]),1))


        # loss = F.binary_cross_entropy(output, new_y, weight = new_w)
        # loss = torch.nn.functional.binary_cross_entropy(out, labels, weight = ww)
        loss = torch.nn.BCELoss(weight=ww)(out, labels)
        # loss = torch.nn.NLLLoss(weight = torch.tensor([0.00001, 1.0])(out, labels))

        # loss = loss + l2reg
        loss = loss
        loss.backward()
        optimizer.step()
        loss_all += loss.item()
    return loss_all


def validate_original(loader, model, device ):
    model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        out = out.view(-1, out.shape[-1])

        labels = torch.tensor(data.y, dtype=torch.float).to(device)
        labels = torch.reshape(labels, (int(list(labels.shape)[0]),1))
        ww = torch.tensor(data.weights, dtype=torch.float).to(device)
        ww = torch.reshape(ww, (int(list(labels.shape)[0]),1))

        # loss = torch.nn.functional.binary_cross_entropy(out, labels, weight = ww)
        loss = torch.nn.BCELoss(weight=ww)(out, labels)
        # loss = torch.nn.NLLLoss(weight = torch.tensor([0.00001, 1.0])(out, labels))

        # loss = loss + l2reg
        loss = loss
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
            out = model(data.x, data.edge_index)
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
    for data in loader:
        #print(len(data))
        if len(data)< 256 : continue
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
        optimizer.step()
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



def validate_edge(loader, model, device, optimizer):
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
