import matplotlib.pyplot as plt
import argparse
import sys
import math
import numpy as np
import pandas as pd
import datetime, os
import torch
from torch_geometric.data import DataListLoader, DataLoader

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
        
# Main function.
'''
def main():


    print('Read data')
    # train dataset
    dataset_train = np.load('data/all_info_df_train.npy')
    x_train = dataset_train[:, :15]
    y_train = dataset_train[:, 15:]

    # val dataset
    dataset_val = np.load('data/all_info_df_val.npy')
    x_val = dataset_val[:, :15]
    y_val = dataset_val[:, 15:]

    # test dataset
    dataset_test = np.load('data/all_info_df_test.npy')
    x_test = dataset_test[:, :15]
    y_test = dataset_test[:, 15:]

    print('Create graphs from data')
    ## create graphs
    graph_list_train = makeGraph(x_train, y_train)
    graph_list_val   = makeGraph(x_val, y_val)
    graph_list_test  = makeGraph(x_test, y_test)

    print('Prepare model')
    ## load model
    model = GATNet_2(14)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print('Prepare optimizer')
    ## define optimizer
    #Adam_weight_decay = 0.001
    learning_ratio = 0.0006
    #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_ratio , weight_decay=Adam_weight_decay )
    optimizer = optim.Adam(model.parameters(), lr=learning_ratio)

    
    print('Prepare DataLoaders')
    ## create DataLoaders
    dataloader_train = DataLoader(graph_list_train, batch_size=512, shuffle=True)
    dataloader_val   = DataLoader(graph_list_val,   batch_size=512, shuffle=True)
    dataloader_test  = DataLoader(graph_list_test,  batch_size=10,  shuffle=True)

    print('Start training')
    train_loss = []
    val_loss   = []
    n_epochs  = 5
    ### train
    for epoch in range(n_epochs):
        print("Epoch:{}".format(epoch+1))
        train_loss.append(train(dataloader_train, model, device, optimizer))
        val_loss.append(validate(dataloader_val, model, device, optimizer))

        print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f},'.format(epoch, train_loss[epoch], val_loss[epoch]))
        torch.save(model.state_dict(), "ckpt/"+'PU_'+"e{:03d}".format(epoch+1)+".pt")
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

'''

def main():
    ## load graphs
    graph_list_train = torch.load('data/graphs_train.pt')
    #graph_list_val   = makeGraph(x_val, y_val)
    graph_list_test  = torch.load('data/graphs_test.pt')
    
    
    ## create validation dataset
    
    size_train = 0.80
    graph_list_val = graph_list_train[int(len(graph_list_train)*size_train) : int(len(graph_list_train))]
    graph_list_train = graph_list_train[0 : int(len(graph_list_train)*size_train) ]
    
    print('Prepare model')
    ## load model
    model = GATNet_2(16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

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
    dataloader_train = DataLoader(graph_list_train, batch_size=512, shuffle=True)
    dataloader_val   = DataLoader(graph_list_val,   batch_size=256, shuffle=True)
    dataloader_test  = DataLoader(graph_list_test,  batch_size=32,  shuffle=True)

    print('Start training')
    train_loss = []
    val_loss   = []
    n_epochs  = 5
    ### train
    os.system('mkdir ckpt')
    
    for epoch in range(n_epochs):
        print("Epoch:{}".format(epoch+1))
        train_loss.append(train(dataloader_train, model, device, optimizer))
        val_loss.append(validate(dataloader_val, model, device, optimizer))

        print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f},'.format(epoch, train_loss[epoch], val_loss[epoch]))
        torch.save(model.state_dict(), "ckpt/"+'PU_'+"e{:03d}".format(epoch+1) + "_losstrain{:.3f}".format(train_loss[epoch]) + "_lossval{:.3f}".format(val_loss[epoch]) + ".pt")
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
