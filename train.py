import matplotlib.pyplot as plt
import argparse
import sys
import math
import numpy as np
import pandas as pd
import datetime, os
import torch
import yaml
from torch_geometric.data import DataListLoader, DataLoader
from torch_geometric.utils import degree

from utils import *
from models import *

def load_yaml(file_name):
    assert(os.path.exists(file_name))
    with open(file_name) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

# Main function.
def main():

    parser = argparse.ArgumentParser(description='Train with configurations')
    add_arg = parser.add_argument
    add_arg('config', help="job configuration")
    args = parser.parse_args()
    config_file = args.config
    config = load_yaml(config_file)

    ## start the code
    path_to_train = config['data']['path_to_train']
    path_to_test = config['data']['path_to_test']

    graph_list_train = torch.load(path_to_train)
    graph_list_test  = torch.load(path_to_test)


    ## create validation dataset // I think is better to create validation dataset here.
    size_train = config['data']['size_train']
    graph_list_val = graph_list_train[int(len(graph_list_train)*size_train) : int(len(graph_list_train))]
    graph_list_train = graph_list_train[0 : int(len(graph_list_train)*size_train) ]

    print('Prepare model')
    choose_model = config['architecture']['choose_model']
    ## load model
    if choose_model == "GATNet":
        # model = GATNet_2(17)
        model = GATNet_2(21) ## added jetRawE, diff Eta
    if choose_model == "EdgeConv":
        # model = EdgeGinNet()
        model = EdgeGinNet(21) ## added jetRawE, diff Eta

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    print('Prepare optimizer')
    ## define optimizer
    learning_ratio = config['architecture']['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=learning_ratio)
    optimizer2 = optim.Adam(model.parameters(), lr=learning_ratio*2)
    optimizer3 = optim.Adam(model.parameters(), lr=learning_ratio*3)


    print('Prepare DataLoaders')
    ## create DataLoaders
    batch_size = config['architecture']['batch_size']
    dataloader_train = DataLoader(graph_list_train, batch_size=batch_size, shuffle=True)
    dataloader_val   = DataLoader(graph_list_val,   batch_size=256, shuffle=True)
    dataloader_test  = DataLoader(graph_list_test,  batch_size=32,  shuffle=True)

    print('Start training')
    train_loss = []
    val_loss   = []
    n_epochs  = config['architecture']['n_epochs']
    ### train
    # os.system('mkdir ckpt')

    path_to_save = config['data']['path_to_save']
    model_name = config['data']['model_name']

    for epoch in range(n_epochs):
        print("Epoch:{}".format(epoch+1))

        train_loss.append(train(dataloader_train, model, device, optimizer))
        val_loss.append(validate(dataloader_val, model, device, optimizer))

        print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f},'.format(epoch, train_loss[epoch], val_loss[epoch]))
        torch.save(model.state_dict(), path_to_save+model_name+"e{:03d}".format(epoch+1) + "_losstrain{:.3f}".format(train_loss[epoch]) + "_lossval{:.3f}".format(val_loss[epoch]) + ".pt")

    plot_ROC_curve(dataloader_val, model, device, "", outdir=path_to_save+model_name)


    return

# Main function call.
if __name__ == '__main__':
    main()
    pass
