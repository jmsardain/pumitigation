import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd


def makeGraph(feature, label):

    ## keep this in mind
    # column_names = ['clusterE', 'clusterEta', 'cluster_CENTER_LAMBDA', 'cluster_CENTER_MAG',
    #             'cluster_ENG_FRAC_EM', 'cluster_FIRST_ENG_DENS', 'cluster_LATERAL', 'cluster_LONGITUDINAL',
    #             'cluster_PTD', 'cluster_time', 'cluster_ISOLATION', 'cluster_SECOND_TIME', 'cluster_SIGNIFICANCE',
    #             'nPrimVtx', 'avgMu']
    graph_list = []
    for i in range(len(feature)):
        num_nodes = len(feature[i])
        edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(i+1, num_nodes)], dtype=torch.long).t().contiguous()

        vec = []

        vec.append(np.array([feature[i][0],  feature[i][1],  feature[i][2],  feature[i][3],  feature[i][4],
                             feature[i][5],  feature[i][6],  feature[i][7],  feature[i][8],  feature[i][9],
                             feature[i][10], feature[i][11], feature[i][12], feature[i][13], feature[i][14]]).T)


        vec = np.array(vec)
        vec = np.squeeze(vec)

        x = torch.tensor(vec, dtype=torch.float)
        w = (np.array(label[i]) > 0.5)*9 + 1


        #graph = Data(x=x, edge_index=edge_index, y=labels[i])
        graph = Data(x=x, edge_index=edge_index, y=torch.tensor(label[i], dtype=torch.float), weights=torch.tensor(w, dtype=torch.float) )
        graph_list.append(graph)

    return graph_list


def train(loader, model, device, optimizer):
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
        loss = torch.nn.functional.binary_cross_entropy(out, labels, weight = ww)
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
        loss_all += loss.item()
    return loss_all

def plot_ROC_curve(loader, model, device):
    model.eval()
    nodes_out = torch.tensor([])
    labels_test = torch.tensor([])
    Cluster_E = torch.tensor([])
    Cluster_Eta = torch.tensor([])
    loss_all = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        out = out.view(-1, out.shape[-1])

        labels = torch.tensor(data.y, dtype=torch.float).to(device)
        labels = torch.reshape(labels, (int(list(labels.shape)[0]),1))
        ww = torch.tensor(data.weights, dtype=torch.float).to(device)
        ww = torch.reshape(ww, (int(list(labels.shape)[0]),1))
        
        Cluster_E_temp = torch.tensor(data.x[:,0], dtype=torch.float).to(device) 
        Cluster_E_temp = torch.reshape(Cluster_E_temp, (int(list(labels.shape)[0]),1))
        Cluster_E = torch.cat((Cluster_E.clone().detach(), Cluster_E_temp.clone().detach()), 0)
        ClusterEta_temp = torch.tensor(data.x[:,1], dtype=torch.float).to(device) 
        ClusterEta_temp = torch.reshape(ClusterEta_temp, (int(list(labels.shape)[0]),1))
        Cluster_Eta = torch.cat((Cluster_Eta.clone().detach(), ClusterEta_temp.clone().detach()), 0)
        
        nodes_out = torch.cat((nodes_out.clone().detach(), out.clone().detach()), 0)
        labels_test = torch.cat((labels_test.clone().detach(), labels.clone().detach()), 0)
        
        loss = torch.nn.functional.binary_cross_entropy(out, labels, weight = ww)        
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
    plt.savefig('roc_curve.png')
    plt.show()
    
def train_edge(loader, model, device, optimizer):
    model.train()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        #out = model(data.x, data.edge_index)
        out = model(data.x, data.edge_index, data.edge_attr)

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