import numpy as np
import pandas as pd
# import torch
import uproot

from utils import *
from models import *
import os
import gc

import torch
from torch_geometric.data import Data
from torch.utils.data import TensorDataset, DataLoader
import pickle
import networkx as nx
from sklearn.model_selection import train_test_split
from utils import *
from models import *


def normalize(x, dic, feature):
    mean, std = np.mean(x), np.std(x)
    out =  (x - mean) / std
    dic[str(feature)] = [mean, std]
    return out, mean, std

def to_networkx(data):
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    edges = data.edge_index.t().tolist()
    G.add_edges_from(edges)
    return G

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



def main():
    # filename = './fracdata.csv'
    # df = pd.read_csv(filename, sep=' ')

    # ### New dataset 60 millions
    # filename = uproot.open('/data/jmsardain/JetCalib/Akt4EMTopo.topo-cluster.root')["ClusterTree"]
    filename = uproot.open('/home/jmsardain/JetCalib/PUMitigation/final/MakeROOT/output_pu.root')["ClusterTree"]
    df = filename.arrays(library="pd")
    # print(len(df))
    # df = df.head(5000000)

    # df = df.head(10)
    ## sort data
    # df = df.sort_values(by=['eventNumber', 'jetCnt'])

    column_names = [
                'clusterE', 'clusterEta', 'cluster_CENTER_LAMBDA', 'cluster_CENTER_MAG',
                'cluster_ENG_FRAC_EM', 'cluster_FIRST_ENG_DENS', 'cluster_LATERAL', 'cluster_LONGITUDINAL',
                'cluster_PTD', 'cluster_time', 'cluster_ISOLATION', 'cluster_SECOND_TIME', 'cluster_SIGNIFICANCE',
                'nPrimVtx', 'avgMu',  'response', ## up until here use these for testing of calibration and add predicted response to graphs!
                'jetCnt', 'clusterPhi', "zL", "zT", "zRel", 'diffEta', 'cluster_nCells',## up to here, we need these features for training
                'eventNumber', 'clusterPt', 'jetCalE', 'jetRawE', 'jetRawPt', 'truthJetE', 'truthJetPt', 'clusterECalib', 'cluster_ENG_CALIB_TOT',  ## add these features to do comprehensive plots at the end
                'labels', ## these are labels, 1 = PU, 0 = signal
                ]

    jetCntValues = np.unique(df.jetCnt.values)
    print(jetCntValues)
    print("total data->",len(df))
    df['diffEta'] = df['clusterEta'] - df['jetCalEta']

    mask = df.cluster_ENG_CALIB_TOT.values != 0
    df['response'] = np.where(mask, np.array( df.clusterE.values ) /  np.array( df.cluster_ENG_CALIB_TOT.values ), -1)
    df['labels'] = ((df['cluster_ENG_CALIB_TOT'] > 0.3) & (df['response'] < 10)).astype(int)  ## 1 means signal, 0 means pileup

    df = df[column_names]


    os.system('mkdir data')
    dic_mean_and_std = {}

    scales = {}
    # log-preprocessing
    field_names = ["clusterE", "cluster_CENTER_LAMBDA", "cluster_FIRST_ENG_DENS",
                   "cluster_SECOND_TIME", "cluster_SIGNIFICANCE",'cluster_CENTER_MAG', "jetRawE"]
    for field_name in field_names:
        x, minimum, epsilon = apply_save_log(df[field_name])
        x, mean, std = normalize(x, dic_mean_and_std, field_name)
        df[field_name] = x

    # just normalizing
    field_names = ["clusterEta", 'clusterPhi', "nPrimVtx", "avgMu"]
    for field_name in field_names:
        x = df[field_name]
        x, mean, std = normalize(x, dic_mean_and_std, field_name)
        df[field_name] = x

    # params between [0, 1]
    # we could also just shift?
    field_names = ["cluster_ENG_FRAC_EM", "cluster_LATERAL", "cluster_LONGITUDINAL",
                   "cluster_PTD", "cluster_ISOLATION", "zL", "zT", "zRel",'diffEta']
    for field_name in field_names:
        x = df[field_name]
        x, mean, std = normalize(x, dic_mean_and_std, field_name)
        df[field_name] = x

    # special preprocessing
    field_name = "cluster_time"
    x = df[field_name]
    x = np.abs(x)**(1./3.) * np.sign(x)
    x, mean, std = normalize(x,dic_mean_and_std, field_name)

    with open('dict_mean_and_std.pkl', 'wb') as f:
        pickle.dump(dic_mean_and_std, f)
    print(dic_mean_and_std)

    df[field_name] = x


    ## get predicted response for each clusters ..
    ## here we should use the dataframe created in this code, using the features from clusterE to avgMu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    testDatasetCalibration = df.to_numpy()
    x_test_calib = testDatasetCalibration[:, 0:15] ## check makeData_calibration for correct index
    y_test_calib = testDatasetCalibration[:, 15]

    x_test_tensor_calib = torch.tensor(x_test_calib, dtype=torch.float32).to(device)
    y_test_tensor_calib = torch.tensor(y_test_calib, dtype=torch.float32).to(device)

    dataset_test_calib = TensorDataset(x_test_tensor_calib, y_test_tensor_calib)
    test_loader_calib = DataLoader(dataset_test_calib, batch_size=4096, shuffle=False)

    num_features = x_test_calib.shape[1]
    predicted_response =  getPredictedResponse(num_features, device, 'ckpts/', test_loader_calib)
    # print(predicted_response)
    df['r_e_prediction'] = predicted_response
    print("\n")
    print(dic_mean_and_std["clusterE"][0])
    print(dic_mean_and_std["clusterE"][1])
    print("\n")
    df['clusterEDNN'] = np.exp(dic_mean_and_std["clusterE"][1]* df['clusterE'] + dic_mean_and_std["clusterE"][0]) / df['r_e_prediction']
    # df.to_csv('./input.csv')
    print("I am here now")

    grouped = df.groupby(['eventNumber', 'jetCnt'])

    graph_list = []

    counter = 1
    for (eventNumber, jetCnt), group_df in grouped:
        if counter%10000==0:
            print("Jet #{}".format(counter))
        counter+=1
        clusterE_list = []
        clusterEta_list = []
        cluster_time_list = []
        cluster_CENTER_LAMBDA_list = []
        cluster_CENTER_MAG_list = []
        cluster_ENG_FRAC_EM_list = []
        cluster_FIRST_ENG_DENS_list = []
        cluster_LATERAL_list = []
        cluster_LONGITUDINAL_list = []
        cluster_PTD_list = []
        cluster_ISOLATION_list = []
        cluster_SECOND_TIME_list = []
        cluster_SIGNIFICANCE_list = []
        nPrimVtx_list = []
        avgMu_list = []
        clusterPhi_list = []
        diffEta_list = []
        zT_list = []
        zL_list = []
        zRel_list = []
        cluster_nCells_list = []
        labels_list = []
        jetRawE_list = []
        jetCalE_list = []
        jetRawPt_list = []
        truthJetE_list = []
        truthJetPt_list = []
        clusterECalib_list = []
        clusterPt_list = []
        cluster_ENG_CALIB_TOT_list = []
        r_e_prediction_list = []
        clusterEDNN_list = []
        eventNumber_list = []

        for index, row in group_df.iterrows():
            clusterE_list.append(row['clusterE'])
            clusterEta_list.append(row['clusterEta'])
            cluster_time_list.append(row['cluster_time'])
            cluster_CENTER_LAMBDA_list.append(row['cluster_CENTER_LAMBDA'])
            cluster_CENTER_MAG_list.append(row['cluster_CENTER_MAG'])
            cluster_ENG_FRAC_EM_list.append(row['cluster_ENG_FRAC_EM'])
            cluster_FIRST_ENG_DENS_list.append(row['cluster_FIRST_ENG_DENS'])
            cluster_LATERAL_list.append(row['cluster_LATERAL'])
            cluster_LONGITUDINAL_list.append(row['cluster_LONGITUDINAL'])
            cluster_PTD_list.append(row['cluster_PTD'])
            cluster_ISOLATION_list.append(row['cluster_ISOLATION'])
            cluster_SECOND_TIME_list.append(row['cluster_SECOND_TIME'])
            cluster_SIGNIFICANCE_list.append(row['cluster_SIGNIFICANCE'])
            nPrimVtx_list.append(row['nPrimVtx'])
            avgMu_list.append(row['avgMu'])
            clusterPhi_list.append(row['clusterPhi'])
            diffEta_list.append(row['diffEta'])
            zT_list.append(row['zT'])
            zL_list.append(row['zL'])
            zRel_list.append(row['zRel'])
            cluster_nCells_list.append(row['cluster_nCells'])
            labels_list.append(row['labels'])
            jetRawE_list.append(row['jetRawE'])
            jetCalE_list.append(row['jetCalE'])
            jetRawPt_list.append(row['jetRawPt'])
            truthJetE_list.append(row['truthJetE'])
            truthJetPt_list.append(row['truthJetPt'])
            clusterECalib_list.append(row['clusterECalib'])
            clusterPt_list.append(row['clusterPt'])
            cluster_ENG_CALIB_TOT_list.append(row['cluster_ENG_CALIB_TOT'])
            r_e_prediction_list.append(row['r_e_prediction'])
            clusterEDNN_list.append(row['clusterEDNN'])
            eventNumber_list.append(row['eventNumber'])

        x = torch.tensor([clusterE_list, clusterEta_list, cluster_time_list,
                          cluster_CENTER_LAMBDA_list, cluster_CENTER_MAG_list, cluster_ENG_FRAC_EM_list,
                          cluster_FIRST_ENG_DENS_list, cluster_LATERAL_list, cluster_LONGITUDINAL_list, cluster_PTD_list,
                          cluster_ISOLATION_list, cluster_SECOND_TIME_list, cluster_SIGNIFICANCE_list, nPrimVtx_list, avgMu_list,
                          clusterPhi_list, diffEta_list, zT_list, zL_list, zRel_list, cluster_nCells_list,
                          jetRawE_list], dtype=torch.float).t()
        edge_index = torch.combinations(torch.arange(len(clusterE_list)), 2).t().contiguous()
        diff_clusterE = torch.abs(x[edge_index[0], 0] - x[edge_index[1], 0])
        diff_clusterEta = torch.abs(x[edge_index[0], 1] - x[edge_index[1], 1])
        diff_clusterPhi = torch.abs(x[edge_index[0], 2] - x[edge_index[1], 2])

        edge_features = torch.stack([diff_clusterE, diff_clusterEta, diff_clusterPhi], dim=1)
        w = np.array(labels_list)
        counts_ones = np.sum(w == 1) ## number of signal clusters
        counts_zeros = np.sum(w == 0) ## number of pu clusters
        w = np.where(w == 0, 0.001, w)
        
        data = Data(x=x, edge_index=edge_index, y=torch.tensor(labels_list, dtype=torch.float), weights=torch.tensor(w, dtype=torch.float))
        data.eventNumber = torch.tensor(labels_list, dtype=torch.float)
        data.jetCnt = jetCnt
        data.jetCalE = jetCalE_list
        data.jetRawPt = jetRawPt_list
        data.truthJetE = truthJetE_list
        data.truthJetPt = truthJetPt_list
        data.clusterECalib = clusterECalib_list
        data.clusterPt = clusterPt_list
        data.cluster_ENG_CALIB_TOT = cluster_ENG_CALIB_TOT_list
        data.r_e_prediction = r_e_prediction_list
        data.clusterEDNN = clusterEDNN_list

        data.edge_attr = edge_features
        graph_list.append(data)


    output_path_graphs = "data/graphs_NewDataset"
    size_train = 0.80
    graphs_test, graph_train = train_test_split(graph_list, test_size = size_train, random_state = 144)
    torch.save(graph_train, output_path_graphs + "_train_new.pt")
    torch.save(graphs_test, output_path_graphs + "_test_new.pt")
    return




# Main function call.
if __name__ == '__main__':
    main()
    pass
