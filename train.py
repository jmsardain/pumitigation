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


# Main function.
def main():


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

    ## create graphs
    graph_list_train = makeGraph(x_train, y_train)
    graph_list_val   = makeGraph(x_val, y_val)
    graph_list_test  = makeGraph(x_test, y_test)

    ## load model
    model = GATNet_2(16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    ## define optimizer
    Adam_weight_decay = 0.001
    learning_ratio = 0.0006
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_ratio , weight_decay=Adam_weight_decay )

    ## create DataLoaders
    dataloader       = DataLoader(graph_list_train, batch_size=512, shuffle=True)
    dataloader_val   = DataLoader(graph_list_val,   batch_size=512, shuffle=True)
    dataloader_test  = DataLoader(graph_list_test,  batch_size=10,  shuffle=True)

    # ## train
    # for epoch in range(n_epochs):
    #     print("Epoch:{}".format(epoch+1))
    #     train_loss.append(train(train_loader, model, device, optimizer))
    #
    #     print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f},'.format(epoch, train_loss[epoch], val_loss[epoch]))
    #     torch.save(model.state_dict(), "ckpt/"+'PU_'+"e{:03d}".format(epoch+1)+".pt")
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
