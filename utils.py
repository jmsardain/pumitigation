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


def train(loader, model, device, optimizer):
    model.train()
    loss_all = 0
    sum1 = aggr.SumAggregation()
    loss_MSE = torch.nn.MSELoss()

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)


        labels = torch.tensor(data.y, dtype=torch.float).to(device)
        jetCnt = data.jetCnt
        jetCnt = torch.tensor(jetCnt, dtype=torch.int64)
        clusterE     = data.x[:,0]
        clusterE    = torch.reshape(clusterE, (int(list(clusterE.shape)[0]),1))
        clusterEDNN  = data.x[:,0] / data.REPredicted
        clusterEDNN = torch.reshape(clusterEDNN, (int(list(clusterEDNN.shape)[0]),1))

        Ejet_target  = sum1(clusterE[labels < 0.1], jetCnt[labels < 0.1] )
        Ejet_train   = sum1(clusterEDNN*out, jetCnt)

        ## reshape now
        labels = torch.reshape(labels, (int(list(labels.shape)[0]),1))
        jetCnt = torch.reshape(jetCnt, (int(list(jetCnt.shape)[0]),1))
        ww = torch.tensor(data.weights, dtype=torch.float).to(device)
        ww = torch.reshape(ww, (int(list(labels.shape)[0]),1))

        loss_reg = loss_MSE(Ejet_train, Ejet_target)
        # Compute focal loss
        # ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(out, labels, reduction='none')
        # ce_loss = torch.nn.functional.binary_cross_entropy(out, labels)
        # p_t = torch.exp(-ce_loss)
        # alpha = 0.25
        # gamma = 0.5
        # loss = (1 - p_t) ** gamma
        # loss = - alpha * loss
        # loss = torch.mean(loss * ce_loss)

        # loss = (alpha * (1 - pt) ** gamma * ce_loss).mean() ## focal loss

        # loss = F.binary_cross_entropy(output, new_y, weight = new_w)
        loss_clf = torch.nn.functional.binary_cross_entropy(out, labels, weight = ww)
        # loss = torch.nn.functional.binary_cross_entropy(out, labels)
        loss = loss_clf + loss_reg
        loss.backward()
        optimizer.step()
        loss_all += loss.item()
    return loss_all


def validate(loader, model, device, optimizer):
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

        loss = torch.nn.functional.binary_cross_entropy(out, labels, weight = ww)
        # loss = torch.nn.functional.binary_cross_entropy(out, labels)

        # ce_loss = torch.nn.functional.binary_cross_entropy(out, labels)
        # p_t = torch.exp(-ce_loss)
        # alpha = 0.25
        # gamma = 0.5
        # loss = (1 - p_t) ** gamma
        # loss = - alpha * loss
        # loss = torch.mean(loss * ce_loss)

        loss_all += loss.item()
    return loss_all

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
