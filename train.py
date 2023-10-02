import matplotlib.pyplot as plt
import argparse
import sys
import math
import numpy as np
import pandas as pd
import datetime, os
import torch
from torch_geometric.data import DataListLoader, DataLoader



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

    return



# Main function call.
if __name__ == '__main__':
    main()
    pass
