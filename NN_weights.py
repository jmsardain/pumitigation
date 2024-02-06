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
import uproot
#import uproot3
from utils import *
from models import *
import yaml
from sklearn.model_selection import train_test_split


def load_yaml(file_name):
    assert(os.path.exists(file_name))
    with open(file_name) as f:
        return yaml.load(f, Loader=yaml.FullLoader)



# Main function.
def main():

    parser = argparse.ArgumentParser(description='Test with configurations')
    add_arg = parser.add_argument
    add_arg('config', help="job configuration")
    args = parser.parse_args()
    config_file = args.config
    config = load_yaml(config_file)


    path_to_train = config['data']['path_to_train']
    graph_list_train = torch.load(path_to_train)

    size_train = config['data']['size_train']
    #graph_list_val = graph_list_train[int(len(graph_list_train)*size_train) : int(len(graph_list_train))]
    #graph_list_train = graph_list_train[0 : int(len(graph_list_train)*size_train) ]
    
    graph_list_val, graph_list_train = train_test_split(graph_list_train, test_size = size_train, random_state = 144)
    
    print('Prepare model')
    choose_model = config['architecture']['choose_model']
    
    #### in case you used some node features ####
    Use_some_Edge_attributes = False

    # if Use_some_Edge_attributes:
    #     graph_list_test  = torch.load('data/graphs_NewDataset_test_data_edges.pt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print('load model')
    # model = GATNet_2(17)
    #model = EdgeGinNet()
    choose_model = config['architecture']['choose_model']
    ## load model
    if choose_model == "GATNet":
        model = GATNet_2(21)
    if choose_model == "EdgeConv":
        model = EdgeGinNet(21)
    
    path_classifier = config['data']['path_classifier']
    model.load_state_dict(torch.load(path_classifier, map_location=torch.device('cpu')))

    model.to(device)

    model_NN = NN_weights()
    model_NN.to(device)

    print('Prepare optimizer')
    ## define optimizer
    learning_ratio = config['architecture']['learning_rate']
    optimizer = optim.Adam(model_NN.parameters(), lr=learning_ratio*1.2)
    optimizer2 = optim.Adam(model_NN.parameters(), lr=learning_ratio*2)
    optimizer3 = optim.Adam(model_NN.parameters(), lr=learning_ratio*3)


    print('Prepare DataLoaders')
    ## create DataLoaders
    batch_size = config['architecture']['batch_size']
    dataloader_train = DataLoader(graph_list_train, batch_size=batch_size, shuffle=True)
    dataloader_val   = DataLoader(graph_list_val,   batch_size=256, shuffle=True)
    #dataloader_test  = DataLoader(graph_list_test,  batch_size=32,  shuffle=True)
    

    print('Start training')
    train_loss = []
    val_loss   = []
    n_epochs  = 30
    
    path_to_save = config['data']['path_to_save']
    model_name = "NN_weights" + config['data']['model_name']

    for epoch in range(n_epochs):
        print("Epoch:{}".format(epoch+1))

        train_loss.append(train_NN_weights(dataloader_train, model_NN, model, device, optimizer) )
        val_loss.append(validate_NN_weights(dataloader_val, model_NN, model, device, optimizer) )

        print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f},'.format(epoch, train_loss[epoch], val_loss[epoch]))
        torch.save(model_NN.state_dict(), path_to_save+model_name+"e{:03d}".format(epoch+1) + "_losstrain{:.3f}".format(train_loss[epoch]) + "_lossval{:.3f}".format(val_loss[epoch]) + ".pt")

    #eerrrooooooooooooorrrrrrrrrrrr
    #plot_ROC_curve(dataloader_val, model, device, "", outdir=path_to_save+model_name)


    return


# Main function call.
if __name__ == '__main__':
    main()
    pass
