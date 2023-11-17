import numpy as np
import pandas as pd
# import torch

from utils import *
from models import *
import os

import torch
from torch_geometric.data import Data

def normalize(x):
    mean, std = np.mean(x), np.std(x)
    out =  (x - mean) / std
    return out, mean, std

def apply_save_log(x):

    #########
    epsilon = 1e-10
    #########

    minimum = x.min()
    if x.min() <= 0:
        x = x - x.min() + epsilon
    else:
        minimum = 0
        epsilon = 0

    return np.log(x), minimum, epsilon

# Main function.

## new main() function
def main():
    filename = './fracdata.csv'
    df = pd.read_csv(filename, sep=' ')
    #df = df.head(4000)

    ## sort data
    df.sort_values(by=['eventNumber', 'jetCnt'])

    column_names = ['clusterE', 'clusterEta', 'cluster_CENTER_LAMBDA', 'cluster_CENTER_MAG',
                'cluster_ENG_FRAC_EM', 'cluster_FIRST_ENG_DENS', 'cluster_LATERAL', 'cluster_LONGITUDINAL',
                'cluster_PTD', 'cluster_time', 'cluster_ISOLATION', 'cluster_SECOND_TIME', 'cluster_SIGNIFICANCE',
                'nPrimVtx', 'avgMu', 'jetCnt', 'clusterPhi', "fracE"]


    df['labels'] = ((df['cluster_ENG_CALIB_TOT'] < 0.0001) & (df['clusterE'] > 0)).astype(int)
    df['fracE'] = df['clusterE'] / df['jetRawE']
    before = df
    df = df[column_names]
    df['labels'] = before['labels']



    os.system('mkdir data')
    #file_path = "all_info_df"
    #output_path_figures_before_preprocessing = "fig.pdf"
    #output_path_figures_after_preprocessing = "fig2.pdf"
    #output_path_data = "data/" + file_path + ".npy"

    #save = True
    #scales_txt_file =  output_path_data[:-4] + "_scales.txt"

    scales = {}
    # log-preprocessing
    field_name = "cluster_time"
    field_names = ["clusterE", "cluster_CENTER_LAMBDA", "cluster_FIRST_ENG_DENS", "cluster_SECOND_TIME", "cluster_SIGNIFICANCE"]
    for field_name in field_names:
        x, minimum, epsilon = apply_save_log(df[field_name])
        x, mean, std = normalize(x)
        df[field_name] = x

    # just normalizing
    field_names = ["clusterEta", "cluster_CENTER_MAG", "nPrimVtx", "avgMu"]
    for field_name in field_names:
        x = df[field_name]
        x, mean, std = normalize(x)
        df[field_name] = x

    # params between [0, 1]
    # we could also just shift?
    field_names = ["cluster_ENG_FRAC_EM", "cluster_LATERAL", "cluster_LONGITUDINAL", "cluster_PTD", "cluster_ISOLATION", "fracE"]
    for field_name in field_names:
        x = df[field_name]
        x, mean, std = normalize(x)
        df[field_name] = x

    # special preprocessing
    field_name = "cluster_time"
    x = df[field_name]
    x = np.abs(x)**(1./3.) * np.sign(x)
    x, mean, std = normalize(x)

    df[field_name] = x

    ### create data per jet
    tempE = []
    tempEta = []
    temp_time = []
    temp_labels = []
    temp_CENTER_LAMBDA = []
    temp_CENTER_MAG = []
    temp_Phi = []
    temp_ENG_FRAC_EM = []
    temp_FIRST_ENG_DENS = []
    temp_LATERAL = []
    temp_LONGITUDINAL = []
    temp_PTD = []
    temp_ISOLATION = []
    temp_SECOND_TIME = []
    temp_SIGNIFICANCE = []
    temp_nPrimVtx = []
    temp_avgMu = []
    temp_fracE = []

    clusterE = []
    clusterEta = []
    cluster_time = []
    labels = []
    cluster_CENTER_LAMBDA = []
    cluster_CENTER_MAG = []
    clusterPhi = []
    cluster_ENG_FRAC_EM = []
    cluster_FIRST_ENG_DENS = []
    cluster_LATERAL = []
    cluster_LONGITUDINAL = []
    cluster_PTD = []
    cluster_ISOLATION = []
    cluster_SECOND_TIME = []
    cluster_SIGNIFICANCE = []
    nPrimVtx = []
    avgMu = []
    fracE = []

    count_jets = 0
    old_event = df['jetCnt'][0]
    old_jetCnt = df['jetCnt'][0]
    change_bol = False


    for index,row in df.iterrows():

        if old_jetCnt != row['jetCnt']:
            old_jetCnt = row['jetCnt']
            count_jets = +1
            if count_jets == 1:
                if len(temp_time)<3:
                    tempE = []
                    tempEta = []
                    temp_time = []
                    temp_labels = []
                    temp_CENTER_LAMBDA = []
                    temp_CENTER_MAG = []
                    temp_Phi = []
                    temp_ENG_FRAC_EM = []
                    temp_FIRST_ENG_DENS = []
                    temp_LATERAL = []
                    temp_LONGITUDINAL = []
                    temp_PTD = []
                    temp_ISOLATION = []
                    temp_SECOND_TIME = []
                    temp_SIGNIFICANCE = []
                    temp_nPrimVtx = []
                    temp_avgMu = []
                    temp_fracE = []
                    continue
                clusterE.append(tempE)
                clusterEta.append(tempEta)
                cluster_time.append(temp_time)
                labels.append(temp_labels)
                #print( len(tempEta) )
                cluster_CENTER_LAMBDA.append(temp_CENTER_LAMBDA)
                cluster_CENTER_MAG.append(temp_CENTER_MAG)
                clusterPhi.append(temp_Phi)

                cluster_ENG_FRAC_EM.append(temp_ENG_FRAC_EM)
                cluster_FIRST_ENG_DENS.append(temp_FIRST_ENG_DENS)
                cluster_LATERAL.append(temp_LATERAL)
                cluster_LONGITUDINAL.append(temp_LONGITUDINAL)
                cluster_PTD.append(temp_PTD)
                cluster_ISOLATION.append(temp_ISOLATION)
                cluster_SECOND_TIME.append(temp_SECOND_TIME)
                cluster_SIGNIFICANCE.append(temp_SIGNIFICANCE)
                nPrimVtx.append(temp_nPrimVtx)
                avgMu.append(temp_avgMu)
                fracE.append(temp_fracE)

                tempE = []
                tempEta = []
                temp_time = []
                temp_labels = []
                temp_CENTER_LAMBDA = []
                temp_CENTER_MAG = []
                temp_Phi = []
                temp_ENG_FRAC_EM = []
                temp_FIRST_ENG_DENS = []
                temp_LATERAL = []
                temp_LONGITUDINAL = []
                temp_PTD = []
                temp_ISOLATION = []
                temp_SECOND_TIME = []
                temp_SIGNIFICANCE = []
                temp_nPrimVtx = []
                temp_avgMu = []
                temp_fracE = []

                count_jets = 0

        tempE.append(row['clusterE'])
        tempEta.append(row['clusterEta'])
        temp_time.append(row['cluster_time'])
        temp_labels.append( float(row['labels'] ) )
        temp_CENTER_LAMBDA.append(row['cluster_CENTER_LAMBDA'])
        temp_CENTER_MAG.append(row['cluster_CENTER_MAG'])
        temp_Phi.append(row['clusterPhi'])
        temp_ENG_FRAC_EM.append(row['cluster_ENG_FRAC_EM'])
        temp_FIRST_ENG_DENS.append(row['cluster_FIRST_ENG_DENS'])
        temp_LATERAL.append(row['cluster_LATERAL'])
        temp_LONGITUDINAL.append(row['cluster_LONGITUDINAL'])
        temp_PTD.append(row['cluster_PTD'])
        temp_ISOLATION.append(row['cluster_ISOLATION'])
        temp_SECOND_TIME.append(row['cluster_SECOND_TIME'])
        temp_SIGNIFICANCE.append(row['cluster_SIGNIFICANCE'])
        temp_nPrimVtx.append(row['nPrimVtx'])
        temp_avgMu.append(row['avgMu'])
        temp_fracE.append(row['fracE'])

    ## create Dictionary containing data and labels
    Dictionary_data ={
        "0": clusterE,
        "1": clusterEta,
        "2": cluster_time,
        "3": cluster_CENTER_LAMBDA,
        "4": cluster_CENTER_MAG,
        "5": cluster_ENG_FRAC_EM ,
        "6": cluster_FIRST_ENG_DENS,
        "7": cluster_LATERAL ,
        "8": cluster_LONGITUDINAL ,
        "9": cluster_PTD ,
        "10": cluster_ISOLATION ,
        "11": cluster_SECOND_TIME,
        "12": cluster_SIGNIFICANCE,
        "13": nPrimVtx,
        "14": avgMu,
        "15": clusterPhi,
        "16": fracE,
        "label": labels
    }


    ## here pytorch data is created
    graph_list = []
    for i in range(len(clusterE)):
        num_nodes = len(clusterE[i])
        edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(i+1, num_nodes)], dtype=torch.long).t().contiguous()
        #print(edge_index)
        vec = []


        vec.append(np.array([Dictionary_data["0"][i], Dictionary_data["1"][i], Dictionary_data["2"][i], Dictionary_data["3"][i], Dictionary_data["4"][i],
                             Dictionary_data["5"][i], Dictionary_data["6"][i], Dictionary_data["7"][i], Dictionary_data["8"][i], Dictionary_data["9"][i],
                             Dictionary_data["10"][i], Dictionary_data["11"][i], Dictionary_data["12"][i], Dictionary_data["13"][i], Dictionary_data["14"][i],
                             Dictionary_data["15"][i], Dictionary_data["16"][i]]).T)

        vec = np.array(vec)
        vec = np.squeeze(vec)

        x = torch.tensor(vec, dtype=torch.float)
        w=(np.array(labels[i]) > 0.5)*9 + 1

        graph = Data(x=x, edge_index=edge_index, y=torch.tensor(labels[i], dtype=torch.float), weights=torch.tensor(w, dtype=torch.float) )

        graph_list.append(graph)


    ## save data
    output_path_graphs = "data/graphs"

    torch.save(graph_list, output_path_graphs + "_fulldata.pt")
    size_train = 0.80
    graphs_test = graph_list[int(len(graph_list)*size_train) : int(len(graph_list))]
    graph_train = graph_list[0 : int(len(graph_list)*size_train) ]

    torch.save(graph_train, output_path_graphs + "_train.pt")
    torch.save(graphs_test, output_path_graphs + "_test.pt")



# Main function call.
if __name__ == '__main__':
    main()
    pass
