import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd

# alpha = 0.25
# gamma = 0.5

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
        loss = torch.nn.functional.binary_cross_entropy(out, labels, weight = ww)
        # loss = torch.nn.functional.binary_cross_entropy(out, labels)
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


from torch_geometric.nn import aggr
import pickle


def train_NN_weights(loader, model_NN, model, device, optimizer):
    model.train()
    loss_all = 0
    loss_all_control = 0
    sum1 = aggr.SumAggregation()
    loss_MSE = torch.nn.MSELoss()
    
    with open('dict_mean_and_std.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

        labels_prev = torch.tensor(data.y, dtype=torch.float).to(device)
        
        labels = torch.reshape(labels_prev, (int(list(labels_prev.shape)[0]),1))
        
        clusterE_prev = data.x[:,0]
        #print("clusterE.size()", clusterE.size())
        #print("test",(torch.ones(len(clusterE))).size() )  
        clusterE = torch.exp( loaded_dict["clusterE"][1]*clusterE_prev + loaded_dict["clusterE"][0]*torch.ones(len(clusterE_prev)).to(device) )
        clusterE = torch.reshape(clusterE, (int(list(clusterE.shape)[0]),1))
        clusterE_copy = clusterE.clone().detach()
        jetCnt_prev = data.jetCnt
        jetCnt_prev = torch.tensor(jetCnt_prev, dtype=torch.int64)
        #print("jetCnt_prev",jetCnt_prev[:60])
        jetCnt = torch.reshape(jetCnt_prev, (int(list(jetCnt_prev.shape)[0]),1))

        #print("clusterE", clusterE[:60] )
        #print("clusterE.size()", clusterE.size())
        #print("labels.size()", labels.size())
        #print(jetCnt.size())

        ## sum over all cluster
        clusterE_sum = (data.clusterE_sum) / 3
        #print(clusterE_sum[:60])
        clusterE_sum = torch.reshape(clusterE_sum, (int(list(clusterE_sum.shape)[0]),1))
        
        clusterE_prev = torch.reshape(clusterE_prev, (int(list(clusterE_prev.shape)[0]),1))

        Jet_E_init = sum1(clusterE_copy, jetCnt_prev )
        Jet_E_inti = Jet_E_init[Jet_E_init != 0]
        Jet_E_input = torch.ones(Jet_E_inti.size())
        
        #in_NN = torch.cat((out.clone().detach(), clusterE_prev.clone().detach()), 1)
        in_NN = torch.cat((out.clone().detach(), clusterE_prev.clone().detach(), clusterE_sum.clone().detach()), 1)
        
        #print(in_NN.size())
        
        out_NN = model_NN( in_NN.to(device) )
        #print("out_NN", out_NN[:70] )
        #print("out_NN.size() ", out_NN.size() )
        #Jet_E = sum1(clusterE[labels], jetCnt_prev[labels] )
        Jet_E = sum1(clusterE[labels_prev < 0.1], jetCnt_prev[labels_prev < 0.1] )
        Jet_NN = sum1(clusterE*out_NN, jetCnt_prev )
        Jet_E = Jet_E[Jet_E != 0]
        Jet_NN = Jet_NN[Jet_NN != 0]

        JetRawE = torch.exp( loaded_dict["jetRawE"][1]*data.JetRawE + loaded_dict["jetRawE"][0]*torch.ones(len(clusterE)).to(device) )
        ##print("JetRawE->", JetRawE[:70] )
        ##print("TruthJetE", data.TruthJetE[:70] )
        #print("Jet_E->", Jet_E[:70] )
        #print("Jet_E.size() ", Jet_E.size() )
        #print("Jet_NN->", Jet_NN[:70] )
        #print("Jet_NN.size() ", Jet_NN.size() )
        ##print("gnn_out->", out.clone().detach()[:70] )
        
        #print("Jet_NN / Jet_E", Jet_NN / Jet_E)
        loss = loss_MSE(Jet_NN / Jet_E, Jet_E / Jet_E)
        loss.backward()
        optimizer.step()
        loss_all += loss.item()
        #print("loss",loss.item())

        ## see how is the loss function taking all the cluster 
        #Jet_E_init = sum1(clusterE_copy, jetCnt_prev )
        #Jet_E_inti = Jet_E_init[Jet_E_init != 0]
        #print("Jet_E.size() ", Jet_E.size() )
        #print("(Jet_E_inti/Jet_E).size() ", (Jet_E_inti/Jet_E).size() )
        #print("Jet_E_inti/Jet_E", Jet_E_inti/Jet_E)
        loss_control = loss_MSE(Jet_E_inti / Jet_E, Jet_E / Jet_E)
        loss_all_control += loss_control.item()
        #print("loss_control",loss_control.item())
    print("loss_NN->", loss_all, "loss_control->", loss_all_control)
    return loss_all



def validate_NN_weights(loader, model_NN, model, device, optimizer):
    model.eval()
    loss_all = 0
    loss_all_control = 0
    sum1 = aggr.SumAggregation()
    loss_MSE = torch.nn.MSELoss()
    with open('dict_mean_and_std.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

        labels_prev = torch.tensor(data.y, dtype=torch.float).to(device)
        labels = torch.reshape(labels_prev, (int(list(labels_prev.shape)[0]),1))
        
        clusterE_prev = data.x[:,0]
        clusterE = torch.exp( loaded_dict["clusterE"][1]*clusterE_prev + loaded_dict["clusterE"][0]*torch.ones(len(clusterE_prev)).to(device) )
        clusterE = torch.reshape(clusterE, (int(list(clusterE.shape)[0]),1))
        clusterE_copy = clusterE.clone().detach()
        jetCnt_prev = data.jetCnt
        jetCnt_prev = torch.tensor(jetCnt_prev, dtype=torch.int64)
        jetCnt = torch.reshape(jetCnt_prev, (int(list(jetCnt_prev.shape)[0]),1))

        ## sum over all cluster
        clusterE_sum = (data.clusterE_sum) / 3
        clusterE_sum = torch.reshape(clusterE_sum, (int(list(clusterE_sum.shape)[0]),1))
        
        clusterE_prev = torch.reshape(clusterE_prev, (int(list(clusterE_prev.shape)[0]),1))
        #in_NN = torch.cat((out.clone().detach(), clusterE_prev.clone().detach()), 1)
        in_NN = torch.cat((out.clone().detach(), clusterE_prev.clone().detach(), clusterE_sum.clone().detach()), 1)
        out_NN = model_NN( in_NN.to(device) )

        Jet_E = sum1(clusterE[labels_prev < 0.1], jetCnt_prev[labels_prev < 0.1] )
        Jet_NN = sum1(clusterE*out_NN, jetCnt_prev )
        Jet_E = Jet_E[Jet_E != 0]
        Jet_NN = Jet_NN[Jet_NN != 0]

        JetRawE = torch.exp( loaded_dict["jetRawE"][1]*data.JetRawE + loaded_dict["jetRawE"][0]*torch.ones(len(clusterE)).to(device) )

        loss = loss_MSE(Jet_NN / Jet_E, Jet_E / Jet_E)
        loss_all += loss.item()

        Jet_E_init = sum1(clusterE_copy, jetCnt_prev )
        Jet_E_inti = Jet_E_init[Jet_E_init != 0]
        
        loss_control = loss_MSE(Jet_E_inti / Jet_E, Jet_E / Jet_E)
        loss_all_control += loss_control.item()
    print("loss_NN validate->", loss_all, "loss_control validate->", loss_all_control)
    return loss_all

    