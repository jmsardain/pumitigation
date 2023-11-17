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

from utils import *
from models import *


## example in case you want to modify entries in the dataset or add atributes
def modify_graphs(graphs):
    new_graphs = []
    for i in range (len(graphs)):
        #graph[i] each graph
        #### example if you want to remove phi and  as node feature
        x_new = graphs[i].x[:,:15]

        graph = Data(x=x_new, edge_index=graphs[i].edge_index, y=torch.tensor(graphs[i].y, dtype=torch.float), weights=torch.tensor(graphs[i].weights, dtype=torch.float) )

        new_graphs.append(graph)

    return new_graphs

##
def modify_graphs_add_EdgesFeatures(graphs):
    new_graphs = []
    for i in range (len(graphs)):
        #graph[i] each graph
        #### example if you want to remove phi and  as node feature
        #x_new = graphs[i].x[:,:15]
        Total_energy = torch.sum(graphs[i].x[:,0])
        Mean_energy = Total_energy / len(graphs[i].x) ## maybe we should use Total_energy
        ## replace phi for Total_energy or Mean_energy
        x_new = graphs[i].x
        x_new[:,15] = Total_energy/4

        edge_attr = []
        Delta_R = []
        Delta_E = []
        for j in range(len(graphs[i].edge_index[0])):
            Delta_eta = graphs[i].x[graphs[i].edge_index[0][j] , 1] - graphs[i].x[graphs[i].edge_index[1][j] , 1]
            Delta_phi = graphs[i].x[graphs[i].edge_index[0][j] , 15] - graphs[i].x[graphs[i].edge_index[1][j] , 15]
            Delta_R_edge = Delta_eta**2 + Delta_phi**2

            Sub_E_rel = np.abs(graphs[i].x[graphs[i].edge_index[0][j]][0] - graphs[i].x[graphs[i].edge_index[1][j]][0])/2

            Delta_R.append(Delta_R_edge*2) # i still need to check if the order is ok
            Delta_E.append(Sub_E_rel)

        edge_attr.append(np.array([Delta_R,Delta_E]).T)
        edge_attr = np.squeeze(np.array(edge_attr))

        graph = Data(x=x_new, edge_index=graphs[i].edge_index, edge_attr=torch.tensor(edge_attr, dtype=torch.float), y=torch.tensor(graphs[i].y, dtype=torch.float), weights=torch.tensor(graphs[i].weights, dtype=torch.float) )

        new_graphs.append(graph)


    return new_graphs

# Main function.
def main():
    ## load graphs
    graph_list_train = torch.load('data/graphs_train.pt')
    #graph_list_val   = makeGraph(x_val, y_val)
    graph_list_test  = torch.load('data/graphs_test.pt')


    #### in case you want to train using some node features ####
    Use_some_Edge_attributes = False
    Do_edge_attributes = False
    '''
    if Use_some_Edge_attributes and Do_edge_attributes:
        output_path_graphs = "data/graphs"
        graph_list_train = modify_graphs_add_EdgesFeatures(graph_list_train)
        torch.save(graph_list_train, output_path_graphs + "_train_data_edges.pt")

        graph_list_test = modify_graphs_add_EdgesFeatures(graph_list_test)
        torch.save(graph_list_test, output_path_graphs + "_test_data_edges.pt")

    if not Do_edge_attributes:
        graph_list_train = torch.load('data/graphs_train_data_edges.pt')
        #graph_list_val   = makeGraph(x_val, y_val)
        graph_list_test  = torch.load('data/graphs_test_data_edges.pt')

    '''
    ## create validation dataset // I think is better to create validation dataset here.

    size_train = 0.80
    graph_list_val = graph_list_train[int(len(graph_list_train)*size_train) : int(len(graph_list_train))]
    graph_list_train = graph_list_train[0 : int(len(graph_list_train)*size_train) ]

    print('Prepare model')
    ## load model
    model = GATNet_2(17)


    # if Use_some_Edge_attributes:
    #     deg = torch.zeros(50, dtype=torch.long)
    #     for data in graph_list_train:
    #         d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    #         #print("d",d)
    #         deg += torch.bincount(d, minlength=deg.numel())
    #
    #     model = PNAConv_EdgeAttrib(16,deg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    #if Use_some_Edge_attributes:
    #    model = PNAConv_EdgeAttrib(16)

    print('Prepare optimizer')
    ## define optimizer
    #Adam_weight_decay = 0.001
    learning_ratio = 0.0006
    #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_ratio , weight_decay=Adam_weight_decay )
    optimizer = optim.Adam(model.parameters(), lr=learning_ratio)
    optimizer2 = optim.Adam(model.parameters(), lr=learning_ratio*2)
    optimizer3 = optim.Adam(model.parameters(), lr=learning_ratio*4)


    print('Prepare DataLoaders')
    ## create DataLoaders
    dataloader_train = DataLoader(graph_list_train, batch_size=1024, shuffle=True)
    dataloader_val   = DataLoader(graph_list_val,   batch_size=256, shuffle=True)
    dataloader_test  = DataLoader(graph_list_test,  batch_size=32,  shuffle=True)

    print('Start training')
    train_loss = []
    val_loss   = []
    n_epochs  = 12
    ### train
    os.system('mkdir ckpt')

    for epoch in range(n_epochs):
        print("Epoch:{}".format(epoch+1))
        if Use_some_Edge_attributes:
            train_loss.append(train_edge(dataloader_train, model, device, optimizer))
            val_loss.append(validate_edge(dataloader_val, model, device, optimizer))
        else:
            train_loss.append(train(dataloader_train, model, device, optimizer))
            val_loss.append(validate(dataloader_val, model, device, optimizer))

        print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f},'.format(epoch, train_loss[epoch], val_loss[epoch]))
        torch.save(model.state_dict(), "ckpt/"+'PU_'+"e{:03d}".format(epoch+1) + "_losstrain{:.3f}".format(train_loss[epoch]) + "_lossval{:.3f}".format(val_loss[epoch]) + ".pt")

    if Use_some_Edge_attributes:
        plot_ROC_curve(dataloader_val, model, device, "edges")
    else:
        plot_ROC_curve(dataloader_val, model, device, "")

    #
    #
    # ## test
    # for epoch in range(n_epochs):
    #     print("Epoch:{}".format(epoch+1))
    #     train_loss.append(train(train_loader, model, device, optimizer))
    #
    #     print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f},'.format(epoch, train_loss[epoch], val_loss[epoch]))
    #     torch.save(model.state_dict(), "ckpt/"+'PU_'+"e{:03d}".format(epoch+1)+".pt")
    #

    return

# Main function call.
if __name__ == '__main__':
    main()
    pass
