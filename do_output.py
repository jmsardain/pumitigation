import matplotlib.pyplot as plt
import argparse
import sys
import math
import numpy as np
import pandas as pd
import datetime, os
import torch
from torch_geometric.data import DataListLoader, DataLoader
from torch_geometric.utils import degree
import pickle 

from utils import *
from models import *


## example in case you want to modify entries in the dataset or add atributes 
def undone_Norm(x, dic_means_stds):
    #out =  (x - mean) / std
    out = x*dic_means_stds[1] + dic_means_stds[0]
    return out
    
# Main function.
def main():
    ## load graphs
    #graph_list_test  = torch.load('data/graphs_test.pt')
    graph_list_test  = torch.load('data/graphs_NewDataset_test.pt')
    
    #### in case you used some node features ####
    Use_some_Edge_attributes = False

    if Use_some_Edge_attributes:
        graph_list_test  = torch.load('data/graphs_NewDataset_test_data_edges.pt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print('load model')
    model = GATNet_2(17)

    path_classifier = "ckpt/PU_batch3000Dropout_Complex_varlr_e015_losstrain125.368_lossval317.697_GATNET.pt"
    #model.load_state_dict(torch.load(path_classifier, map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(path_classifier))
    
    if Use_some_Edge_attributes:
        deg = torch.zeros(50, dtype=torch.long)
        for data in graph_list_train:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            #print("d",d)
            deg += torch.bincount(d, minlength=deg.numel())
        model = PNAConv_EdgeAttrib(16,deg)
    
    model.to(device)   

    #print('Prepare DataLoaders')
    ## create DataLoaders
    dataloader_test  = DataLoader(graph_list_test,  batch_size=1024,  shuffle=False) # please use batch_size=1, if not jet count will be wrong
    
    os.system('mkdir ouput_dataframes')
    
    #print('doing scores ')
    model.eval()
    nodes_out = torch.tensor([])
    labels_test = torch.tensor([])
    Cluster_E = torch.tensor([])
    Cluster_Eta = torch.tensor([])
    jetCnt = torch.tensor([])
    jet_count = 0
    jetRawE = torch.tensor([])

    # clusterE  jetRawE jetCnt score 

    with torch.no_grad():
        #for data in graph_list_test:
        for data in dataloader_test:
            data = data.to(device)
    
            if Use_some_Edge_attributes:
                out = model(data.x, data.edge_index, data.edge_attr)
            else:
                out = model(data.x, data.edge_index)
            out = out.view(-1, out.shape[-1])
    
            #data = data 
            labels = torch.tensor(data.y , dtype=torch.float).cpu()
            labels = torch.reshape(labels, (int(list(labels.shape)[0]),1))
            ww = torch.tensor(data.weights , dtype=torch.float).cpu()
            ww = torch.reshape(ww, (int(list(labels.shape)[0]),1))
            
            Cluster_E_temp = torch.tensor(data.x[:,0] , dtype=torch.float).cpu()#to(device) 
            Cluster_E_temp = torch.reshape(Cluster_E_temp, (int(list(labels.shape)[0]),1))
            Cluster_E = torch.cat((Cluster_E.clone().detach(), Cluster_E_temp.clone().detach().cpu()), 0)
            #ClusterEta_temp = torch.tensor(data.x[:,1] , dtype=torch.float).cpu()#.to(device) 
            #ClusterEta_temp = torch.reshape(ClusterEta_temp, (int(list(labels.shape)[0]),1))
            #Cluster_Eta = torch.cat((Cluster_Eta.clone().detach(), ClusterEta_temp.clone().detach().cpu()), 0)
    
            nodes_out = torch.cat((nodes_out.clone().detach(), out.clone().detach().cpu()), 0)
            labels_test = torch.cat((labels_test.clone().detach(), labels.clone().detach().cpu()), 0)
    
            jet_count += len(data)
            print(jet_count)
            #jetCnt_temp = torch.ones(len(data))*jet_count
            jetCnt_temp = torch.tensor(data.jetCnt , dtype=torch.int).cpu()#to(device) 
            jetCnt_temp = torch.reshape(jetCnt_temp, (int(list(jetCnt_temp.shape)[0]),1))
            jetCnt = torch.cat((jetCnt.clone().detach(), jetCnt_temp.clone().detach().cpu()), 0)
          
            jetRawE_temp = torch.tensor(data.JetRawE , dtype=torch.float).cpu()#to(device) 
            jetRawE_temp = torch.reshape(jetRawE_temp, (int(list(jetRawE_temp.shape)[0]),1))
            jetRawE = torch.cat((jetRawE.clone().detach(), jetRawE_temp.clone().detach().cpu()), 0)
            
            print("jetRawE:",len(jetRawE) ," jetCnt:",len(jetCnt)," nodes_out:",len(nodes_out)  )

            #if jet_count>2000: break
            
    nodes_out = torch.squeeze(nodes_out)
    labels_test = torch.squeeze(labels_test)
    Cluster_E = torch.squeeze(Cluster_E)
    jetRawE = torch.squeeze(jetRawE)
    df_out = pd.DataFrame()
    df_out['labels'] = labels_test
    df_out['score'] = nodes_out
    df_out['clusterE'] = Cluster_E
    #df_out['ClusterEta'] = Cluster_Eta
    df_out['jetCnt'] = jetCnt
    df_out['JetRawE'] = jetRawE

    ## now is necesary to undone the normalization
    # load dictionary with means and std
    with open('dict_mean_and_std.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    norm_variables = ['clusterE'] # ['ClusterE','ClusterEta']
    for field_name in norm_variables:
        df_out[field_name] = undone_Norm(df_out[field_name], loaded_dict[field_name])
    
    #df.to_csv(index=False)
    df_out.to_csv('./ouput_dataframes/out.csv')  
    
    print(df_out)
    
    return

# Main function call.
if __name__ == '__main__':
    main()
    pass
