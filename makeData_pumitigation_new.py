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


def get_pt_weight(pt_value):
    pt_value_np = np.array(pt_value)
    pt_ranges = np.array([(0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0), (20.0, 25.0), (25.0, 30.0), (30.0, 35.0), (35.0, 40.0), (40.0, 45.0), (45.0, 50.0), (50.0, 55.0), (55.0, 60.0), (60.0, 65.0), (65.0, 70.0), (70.0, 75.0), (75.0, 80.0), (80.0, 85.0), (85.0, 90.0), (90.0, 95.0), (95.0, 100.0), (100.0, 105.0), (105.0, 110.0), (110.0, 115.0), (115.0, 120.0), (120.0, 125.0), (125.0, 130.0), (130.0, 135.0), (135.0, 140.0), (140.0, 145.0), (145.0, 150.0), (150.0, 155.0), (155.0, 160.0), (160.0, 165.0), (165.0, 170.0), (170.0, 175.0), (175.0, 180.0), (180.0, 185.0), (185.0, 190.0), (190.0, 195.0), (195.0, 200.0), (200.0, 205.0), (205.0, 210.0), (210.0, 215.0), (215.0, 220.0), (220.0, 225.0), (225.0, 230.0), (230.0, 235.0), (235.0, 240.0), (240.0, 245.0), (245.0, 250.0), (250.0, 255.0), (255.0, 260.0), (260.0, 265.0), (265.0, 270.0), (270.0, 275.0), (275.0, 280.0), (280.0, 285.0), (285.0, 290.0), (290.0, 295.0), (295.0, 300.0), (300.0, 305.0), (305.0, 310.0), (310.0, 315.0), (315.0, 320.0), (320.0, 325.0), (325.0, 330.0), (330.0, 335.0), (335.0, 340.0), (340.0, 345.0), (345.0, 350.0), (350.0, 355.0), (355.0, 360.0), (360.0, 365.0), (365.0, 370.0), (370.0, 375.0), (375.0, 380.0), (380.0, 385.0), (385.0, 390.0), (390.0, 395.0), (395.0, 400.0), (400.0, 405.0), (405.0, 410.0), (410.0, 415.0), (415.0, 420.0), (420.0, 425.0), (425.0, 430.0), (430.0, 435.0), (435.0, 440.0), (440.0, 445.0), (445.0, 450.0), (450.0, 455.0), (455.0, 460.0), (460.0, 465.0), (465.0, 470.0), (470.0, 475.0), (475.0, 480.0), (480.0, 485.0), (485.0, 490.0), (490.0, 495.0), (495.0, 500.0), ])

    # pt_weights = np.array([1107.0, 1524877.0, 6196985.0, 7575628.0, 5248034.0, 3462525.0, 2595018.0, 2211429.0, 1979934.0, 1810449.0, 1658343.0, 1493440.0, 1357220.0, 1239234.0, 1125966.0, 1038119.0, 977983.0, 918803.0, 870618.0, 824128.0, 791393.0, 771152.0, 747034.0, 727981.0, 706840.0, 673288.0, 650053.0, 613848.0, 586398.0, 553826.0, 542058.0, 503923.0, 477101.0, 456459.0, 437361.0, 415444.0, 391903.0, 373599.0, 356798.0, 340946.0, 326174.0, 308659.0, 297948.0, 281489.0, 272320.0, 256260.0, 241885.0, 228606.0, 219627.0, 207244.0, 193493.0, 182360.0, 167810.0, 158317.0, 146020.0, 132194.0, 117740.0, 106575.0, 92808.0, 79139.0, 66202.0, 54099.0, 44240.0, 34888.0, 27275.0, 20125.0, 13447.0, 9540.0, 7390.0, 4607.0, 3779.0, 2096.0, 1321.0, 841.0, 444.0, 202.0, 167.0, 57.0, 61.0, 19.0, 35.0, 0.0, 25.0, 0.0, 0.0, 19.0, 17.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ])

    pt_weights = np.array([1.859416396212844e-05, 0.025613200505942665, 0.10409011306309893, 0.1272470362674717, 0.08815068172974233, 0.058159672604307835, 0.04358824767540616, 0.03714514310443156, 0.033256745646063963, 0.030409923713705034, 0.027855015038345043, 0.025085156483831163, 0.02279708329962056, 0.020815284718558515, 0.01891273389320859, 0.017437176963144365, 0.016427078820392283, 0.015433038510294036, 0.014623680072610964, 0.013842792373785896, 0.013292945980560716, 0.012952960006976757, 0.012547852467285146, 0.01222782120624591, 0.011872718026188677, 0.01130914856886498, 0.01091887268841326, 0.010310741065785563, 0.009849666268350672, 0.009302557769186762, 0.009104891896100649, 0.008464342264035817, 0.008013815917340055, 0.007667094388427448, 0.00734630726706455, 0.0069781696956481365, 0.006582753964995503, 0.0062753035791212495, 0.005993098928057365, 0.0057268345313747455, 0.005478710782460055, 0.005184513147594039, 0.0050046015936659844, 0.0047281414810619446, 0.004574130740891433, 0.004304372589823879, 0.004062917208653512, 0.0038398712255883777, 0.003689051895673336, 0.0034810559314971514, 0.0032500818134815882, 0.0030630819694071744, 0.0028186871314225597, 0.0026592341969216694, 0.002452682765808487, 0.0022204488805868173, 0.0019776665446260186, 0.0017901291998769996, 0.0015588863315241339, 0.001329289558987247, 0.0011119881143819574, 0.0009086952811085695, 0.0007430946826418809, 0.0005860101104884707, 0.00045813534062064424, 0.0003380375336385139, 0.00022586786160681222, 0.00016024238861671663, 0.00012412906204167044, 7.738330024708738e-05, 6.347547029167423e-05, 3.520629418664969e-05, 2.2188699723551644e-05, 1.4126189604471561e-05, 7.457821860149076e-06, 3.392973008446201e-06, 2.805081645596612e-06, 9.574230766407597e-07, 1.0246106609664271e-06, 3.191410255469199e-07, 5.878913628495893e-07, 0.0, 4.1992240203542095e-07, 0.0, 0.0, 3.191410255469199e-07, 2.855472333840862e-07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ])

    masks = [(pt_value_np >= pt_min) & (pt_value_np < pt_max) for pt_min, pt_max in pt_ranges]
    pt_weight = pt_weights[np.argmax(masks, axis=0)]

    return pt_weight

def main():
    # filename = './fracdata.csv'
    # df = pd.read_csv(filename, sep=' ')

    # ### New dataset 60 millions
    # filename = uproot.open('/data/jmsardain/JetCalib/Akt4EMTopo.topo-cluster.root')["ClusterTree"]
    # filename = uproot.open('/home/jmsardain/JetCalib/PUMitigation/final/MakeROOT/output_pu.root')["ClusterTree"]
    filename = uproot.open('/home/jmsardain/JetCalib/PUMitigation/final/Resample/output_pu.root')["ClusterTree"]
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
                'jetAreaE', 'jetAreaPt', 'labels', 'clusterE_original' ## these are labels, 1 = PU, 0 = signal
                ]

    jetCntValues = np.unique(df.jetCnt.values)
    print(jetCntValues)
    print("total data->",len(df))
    df['diffEta'] = df['clusterEta'] - df['jetCalEta']

    df['response'] = np.where( df.cluster_ENG_CALIB_TOT.values != 0 , np.array( df.clusterE.values ) /  np.array( df.cluster_ENG_CALIB_TOT.values ), 100)
    # df['labels'] = ((df['cluster_ENG_CALIB_TOT'] > 0.3) & (df['response'] < 4)).astype(int)  ## 1 means signal, 0 means pileup
    df['labels'] = ((df['cluster_ENG_CALIB_TOT'] > 0.3) & (df['response'] > 0.2) & (df['response'] < 4)).astype(int)  ## 1 means signal, 0 means pileup
    df['clusterE_original'] = df['clusterE']
    df = df[column_names]

    os.system('mkdir data')
    dic_mean_and_std = {}

    scales = {}
    # log-preprocessing
    field_names = ["clusterE", "cluster_CENTER_LAMBDA", "cluster_FIRST_ENG_DENS",
                   "cluster_SECOND_TIME", "cluster_SIGNIFICANCE",'cluster_CENTER_MAG']
    for field_name in field_names:
        x, minimum, epsilon = apply_save_log(df[field_name])
        x, mean, std = normalize(x, dic_mean_and_std, field_name)
        df[field_name] = x

    # just normalizing
    # field_names = ["clusterEta", 'clusterPhi', "nPrimVtx", "avgMu"]
    # for field_name in field_names:
    #     x = df[field_name]
    #     x, mean, std = normalize(x, dic_mean_and_std, field_name)
    #     df[field_name] = x

    # params between [0, 1]
    # we could also just shift?
    # field_names = ["cluster_ENG_FRAC_EM", "cluster_LATERAL", "cluster_LONGITUDINAL",
    #                "cluster_PTD", "cluster_ISOLATION", "zL", "zT", "zRel",'diffEta']
    # for field_name in field_names:
    #     x = df[field_name]
    #     x, mean, std = normalize(x, dic_mean_and_std, field_name)
    #     df[field_name] = x

    # special preprocessing
    # field_name = "cluster_time"
    # x = df[field_name]
    # x = np.abs(x)**(1./3.) * np.sign(x)
    # x, mean, std = normalize(x,dic_mean_and_std, field_name)

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
    # print("\n")
    # print(dic_mean_and_std["clusterE"][0])
    # print(dic_mean_and_std["clusterE"][1])
    # print("\n")
    # df['clusterEDNN'] = np.exp(dic_mean_and_std["clusterE"][1]* df['clusterE'] + dic_mean_and_std["clusterE"][0]) / df['r_e_prediction']
    # # df.to_csv('./input.csv')
    df['clusterEDNN'] = df['clusterE']
    print("I am here now")

    grouped = df.groupby(['eventNumber', 'jetCnt'])

    graph_list = []

    counter = 1
    for (eventNumber, jetCnt), group_df in grouped:
        # if counter>5: continue
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
        jetCnt_list = []
        jetAreaE_list = []
        jetAreaPt_list = []
        clusterE_original_list = []

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
            jetCnt_list.append(row['jetCnt'])
            jetAreaE_list.append(row['jetAreaE'])
            jetAreaPt_list.append(row['jetAreaPt'])
            clusterE_original_list.append(row['clusterE_original'])

        x = torch.tensor([clusterE_list, clusterEta_list, cluster_time_list,
                          cluster_CENTER_LAMBDA_list, cluster_CENTER_MAG_list, cluster_ENG_FRAC_EM_list,
                          cluster_FIRST_ENG_DENS_list, cluster_LATERAL_list, cluster_LONGITUDINAL_list, cluster_PTD_list,
                          cluster_ISOLATION_list, cluster_SECOND_TIME_list, cluster_SIGNIFICANCE_list, nPrimVtx_list, avgMu_list,
                          clusterPhi_list, diffEta_list, zT_list, zL_list, zRel_list, cluster_nCells_list,
                          jetRawE_list], dtype=torch.float).t()


        edge_index = torch.combinations(torch.arange(len(clusterE_list)), 2).t().contiguous()
        diff_clusterE = torch.abs(x[edge_index[0], 0] - x[edge_index[1], 0])
        diff_clusterEta = torch.abs(x[edge_index[0], 1] - x[edge_index[1], 1])
        diff_cluster_time = torch.abs(x[edge_index[0], 2] - x[edge_index[1], 2])
        diff_cluster_CENTER_LAMBDA = torch.abs(x[edge_index[0], 3] - x[edge_index[1], 3])
        diff_cluster_CENTER_MAG = torch.abs(x[edge_index[0], 4] - x[edge_index[1], 4])
        diff_cluster_ENG_FRAC_EM = torch.abs(x[edge_index[0], 5] - x[edge_index[1], 5])
        diff_cluster_FIRST_ENG_DENS = torch.abs(x[edge_index[0], 6] - x[edge_index[1], 6])
        diff_cluster_LATERAL = torch.abs(x[edge_index[0], 7] - x[edge_index[1], 7])
        diff_cluster_LONGITUDINAL = torch.abs(x[edge_index[0], 8] - x[edge_index[1], 8])
        diff_cluster_PTD = torch.abs(x[edge_index[0], 9] - x[edge_index[1], 9])
        diff_cluster_ISOLATION = torch.abs(x[edge_index[0], 10] - x[edge_index[1], 10])
        diff_cluster_SECOND_TIME = torch.abs(x[edge_index[0], 11] - x[edge_index[1], 11])
        diff_cluster_SIGNIFICANCE = torch.abs(x[edge_index[0], 12] - x[edge_index[1], 12])
        diff_nPrimVtx = torch.abs(x[edge_index[0], 13] - x[edge_index[1], 13])
        diff_avgMu = torch.abs(x[edge_index[0], 14] - x[edge_index[1], 14])
        diff_clusterPhi = torch.abs(x[edge_index[0], 15] - x[edge_index[1], 15])
        diff_diffEta = torch.abs(x[edge_index[0], 16] - x[edge_index[1], 16])
        diff_zT = torch.abs(x[edge_index[0], 17] - x[edge_index[1], 17])
        diff_zL = torch.abs(x[edge_index[0], 18] - x[edge_index[1], 18])
        diff_zRel = torch.abs(x[edge_index[0], 19] - x[edge_index[1], 19])
        diff_cluster_nCells = torch.abs(x[edge_index[0], 20] - x[edge_index[1], 20])
        diff_jetRawE = torch.abs(x[edge_index[0], 21] - x[edge_index[1], 21])

        # edge_features = torch.stack([diff_clusterE, diff_clusterEta, diff_clusterPhi], dim=1)
        edge_features = torch.stack([diff_clusterE, diff_clusterEta, diff_cluster_time,
                                     diff_cluster_CENTER_LAMBDA, diff_cluster_CENTER_MAG, diff_cluster_ENG_FRAC_EM,
                                     diff_cluster_FIRST_ENG_DENS, diff_cluster_LATERAL, diff_cluster_LONGITUDINAL, diff_cluster_PTD,
                                     diff_cluster_ISOLATION, diff_cluster_SECOND_TIME, diff_cluster_SIGNIFICANCE, diff_nPrimVtx, diff_avgMu,
                                     diff_clusterPhi, diff_diffEta, diff_zT, diff_zL, diff_zRel, diff_cluster_nCells,
                                     diff_jetRawE], dim=1)
        w = np.array(labels_list)
        counts_ones = np.sum(w == 1) ## number of signal clusters
        counts_zeros = np.sum(w == 0) ## number of pu clusters

        # NewWeight = 0
        if counts_zeros>0 and counts_ones>0:
            NewWeight = counts_ones/ counts_zeros
        else:
            NewWeight = 1.0
        w = np.where(w == 0 , NewWeight, w)

        # print(labels_list)
        # print(jetCnt_list)
        # print(jetRawE_list)
        w_pt = get_pt_weight(jetRawPt_list[0])

        data = Data(x=x, edge_index=edge_index, y=torch.tensor(labels_list, dtype=torch.float), weights=torch.tensor(w, dtype=torch.float))
        data.eventNumber = torch.tensor(eventNumber_list, dtype=torch.float)
        data.jetCnt = torch.tensor(jetCnt_list, dtype=torch.float)
        data.JetCalE = torch.tensor(jetCalE_list, dtype=torch.float)
        data.JetRawE = torch.tensor(jetRawE_list, dtype=torch.float)
        data.JetRawPt = torch.tensor(jetRawPt_list, dtype=torch.float)
        data.TruthJetE = torch.tensor(truthJetE_list, dtype=torch.float)
        data.TruthJetPt = torch.tensor(truthJetPt_list, dtype=torch.float)
        data.ClusterECalib = torch.tensor(clusterECalib_list, dtype=torch.float)
        data.ClusterPt = torch.tensor(clusterPt_list, dtype=torch.float)
        data.ClusterENGCALIBTOT = torch.tensor(cluster_ENG_CALIB_TOT_list, dtype=torch.float)
        data.REPredicted = torch.tensor(r_e_prediction_list, dtype=torch.float)
        data.clusterEDNN = torch.tensor(clusterEDNN_list, dtype=torch.float)
        data.ClusterEOriginal = torch.tensor(clusterE_original_list, dtype=torch.float)
        data.JetAreaE = torch.tensor(jetAreaE_list, dtype=torch.float)
        data.JetAreaPt = torch.tensor(jetAreaPt_list, dtype=torch.float)
        data.pt_weight = torch.tensor(w_pt, dtype=torch.float)
        edge_attr=torch.tensor( edge_features , dtype=torch.float)
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
