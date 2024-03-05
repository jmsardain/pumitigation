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
import uproot3
from utils import *
from models import *
import yaml

def load_yaml(file_name):
    assert(os.path.exists(file_name))
    with open(file_name) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


## example in case you want to modify entries in the dataset or add atributes
def undone_Norm(x, dic_means_stds):
    #out =  (x - mean) / std
    out = x*dic_means_stds[1] + dic_means_stds[0]
    return out

# Main function.
def main():

    parser = argparse.ArgumentParser(description='Test with configurations')
    add_arg = parser.add_argument
    add_arg('config', help="job configuration")
    args = parser.parse_args()
    config_file = args.config
    config = load_yaml(config_file)

    ## load graphs
    path_to_test = config['data']['path_to_test']
    #graph_list_test  = torch.load('data/graphs_test.pt')
    graph_list_test  = torch.load('data/graphs_NewDataset_test.pt')

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

    # path_classifier = "ckpt/PU_batch3000Dropout_Complex_varlr_e015_losstrain125.368_lossval317.697_GATNET.pt"
    # path_classifier = "ckpt/PU_batch3000Dropout_Complex_varlr_GATConv_e015_losstrain49.022_lossval38.814.pt"
    # path_classifier = "ckpt/PU_batch3000Dropout_Complex_varlr_GATConv_e015_losstrain48.585_lossval39.881.pt"
    # path_classifier = "ckpt/PU_batch3000Dropout_Complex_varlr_GATConv_e015_losstrain48.873_lossval40.086.pt"
    # path_classifier = "ckpt/PU_batch3000Dropout_Complex_varlr_GATConv_e014_losstrain13.928_lossval9.873.pt"
    # path_classifier = "ckpt/PU_batch3000Dropout_Complex_varlr_GATConv_e011_losstrain51.235_lossval29.801.pt"
    # path_classifier = "ckpt/PU_batch3000Dropout_Complex_varlr_GATConv_e011_losstrain49.289_lossval26.954.pt"
    # path_classifier = "ckpt/PU_batch3000Dropout_Complex_varlr_GATConv_e100_losstrain44.974_lossval48.839.pt"
    path_classifier = config['data']['path_classifier']
    #model.load_state_dict(torch.load(path_classifier, map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(path_classifier))

    # if Use_some_Edge_attributes:
    #     deg = torch.zeros(50, dtype=torch.long)
    #     for data in graph_list_train:
    #         d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    #         #print("d",d)
    #         deg += torch.bincount(d, minlength=deg.numel())
    #     model = PNAConv_EdgeAttrib(16,deg)

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
    Cluster_Phi = torch.tensor([])
    
    jetCnt = torch.tensor([])
    eventNumber = torch.tensor([])

    jet_count = 0
    ## 'jetRawE', 'jetRawPt', 'truthJetE', 'truthJetPt', 'clusterECalib', 'ClusterENGCALIBTOT'
    jetRawE = torch.tensor([])
    jetCalE = torch.tensor([])
    jetRawPt = torch.tensor([])
    truthJetE = torch.tensor([])
    truthJetPt = torch.tensor([])
    clusterECalib = torch.tensor([])
    ClusterPt = torch.tensor([])
    ClusterENGCALIBTOT = torch.tensor([])
    r_e_predicted = torch.tensor([])


    # clusterE  jetRawE jetCnt score

    with torch.no_grad():
        #for data in graph_list_test:
        for data in dataloader_test:
            data = data.to(device)

            # if Use_some_Edge_attributes:
            #     out = model(data.x, data.edge_index, data.edge_attr)
            # else:
            #     out = model(data.x, data.edge_index)
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

            ## get clusterEta
            Cluster_Eta_temp = torch.tensor(data.x[:,1] , dtype=torch.float).cpu()#to(device)
            Cluster_Eta_temp = torch.reshape(Cluster_Eta_temp, (int(list(labels.shape)[0]),1))
            Cluster_Eta = torch.cat((Cluster_Eta.clone().detach(), Cluster_Eta_temp.clone().detach().cpu()), 0)
            ## get clusterPhi
            Cluster_Phi_temp = torch.tensor(data.x[:,15] , dtype=torch.float).cpu()#to(device)
            Cluster_Phi_temp = torch.reshape(Cluster_Phi_temp, (int(list(labels.shape)[0]),1))
            Cluster_Phi = torch.cat((Cluster_Phi.clone().detach(), Cluster_Phi_temp.clone().detach().cpu()), 0)


            nodes_out = torch.cat((nodes_out.clone().detach(), out.clone().detach().cpu()), 0)
            labels_test = torch.cat((labels_test.clone().detach(), labels.clone().detach().cpu()), 0)

            jet_count += len(data)
            print(jet_count)
            #jetCnt_temp = torch.ones(len(data))*jet_count
            jetCnt_temp = torch.tensor(data.jetCnt , dtype=torch.int).cpu()#to(device)
            jetCnt_temp = torch.reshape(jetCnt_temp, (int(list(jetCnt_temp.shape)[0]),1))
            jetCnt = torch.cat((jetCnt.clone().detach(), jetCnt_temp.clone().detach().cpu()), 0)


            ## eventNumber
            eventNumber_temp = torch.tensor(data.eventNumber , dtype=torch.int).cpu()#to(device)
            eventNumber_temp = torch.reshape(eventNumber_temp, (int(list(eventNumber_temp.shape)[0]),1))
            eventNumber = torch.cat((eventNumber.clone().detach(), eventNumber_temp.clone().detach().cpu()), 0)

            ## jetRawE
            jetRawE_temp = torch.tensor(data.JetRawE , dtype=torch.float).cpu()#to(device)
            jetRawE_temp = torch.reshape(jetRawE_temp, (int(list(jetRawE_temp.shape)[0]),1))
            jetRawE = torch.cat((jetRawE.clone().detach(), jetRawE_temp.clone().detach().cpu()), 0)

            ## 'jetRawPt'
            jetRawPt_temp = torch.tensor(data.JetRawPt , dtype=torch.float).cpu()#to(device)
            jetRawPt_temp = torch.reshape(jetRawPt_temp, (int(list(jetRawPt_temp.shape)[0]),1))
            jetRawPt = torch.cat((jetRawPt.clone().detach(), jetRawPt_temp.clone().detach().cpu()), 0)

            ## 'truthJetE'
            truthJetE_temp = torch.tensor(data.TruthJetE , dtype=torch.float).cpu()#to(device)
            truthJetE_temp = torch.reshape(truthJetE_temp, (int(list(truthJetE_temp.shape)[0]),1))
            truthJetE = torch.cat((truthJetE.clone().detach(), truthJetE_temp.clone().detach().cpu()), 0)

            ## 'truthJetPt'
            truthJetPt_temp = torch.tensor(data.TruthJetPt , dtype=torch.float).cpu()#to(device)
            truthJetPt_temp = torch.reshape(truthJetPt_temp, (int(list(truthJetPt_temp.shape)[0]),1))
            truthJetPt = torch.cat((truthJetPt.clone().detach(), truthJetPt_temp.clone().detach().cpu()), 0)

            ## 'clusterECalib'
            clusterECalib_temp = torch.tensor(data.ClusterECalib , dtype=torch.float).cpu()#to(device)
            clusterECalib_temp = torch.reshape(clusterECalib_temp, (int(list(clusterECalib_temp.shape)[0]),1))
            clusterECalib = torch.cat((clusterECalib.clone().detach(), clusterECalib_temp.clone().detach().cpu()), 0)

            ## 'clusterENGCalibTOT'
            clusterENGCalibTOT_temp = torch.tensor(data.ClusterENGCALIBTOT , dtype=torch.float).cpu()#to(device)
            clusterENGCalibTOT_temp = torch.reshape(clusterENGCalibTOT_temp, (int(list(clusterENGCalibTOT_temp.shape)[0]),1))
            ClusterENGCALIBTOT = torch.cat((ClusterENGCALIBTOT.clone().detach(), clusterENGCalibTOT_temp.clone().detach().cpu()), 0)

            ## clusterPt
            clusterPt_temp = torch.tensor(data.ClusterPt , dtype=torch.float).cpu()#to(device)
            clusterPt_temp = torch.reshape(clusterPt_temp, (int(list(clusterPt_temp.shape)[0]),1))
            ClusterPt = torch.cat((ClusterPt.clone().detach(), clusterPt_temp.clone().detach().cpu()), 0)
            
            ## 'jetCalE'
            jetCalE_temp = torch.tensor(data.JetCalE , dtype=torch.float).cpu()#to(device)
            jetCalE_temp = torch.reshape(jetCalE_temp, (int(list(jetCalE_temp.shape)[0]),1))
            jetCalE = torch.cat((jetCalE.clone().detach(), jetCalE_temp.clone().detach().cpu()), 0)

            


            ## predicted response
            r_e_predicted_temp = torch.tensor(data.REPredicted , dtype=torch.float).cpu()#to(device)
            r_e_predicted_temp = torch.reshape(r_e_predicted_temp, (int(list(r_e_predicted_temp.shape)[0]),1))
            r_e_predicted = torch.cat((r_e_predicted.clone().detach(), r_e_predicted_temp.clone().detach().cpu()), 0)


            # print("jetRawE:",len(jetRawE) ," jetCnt:",len(jetCnt)," nodes_out:",len(nodes_out)  )

            #if jet_count>2000: break

    jetCnt = torch.squeeze(jetCnt)
    eventNumber = torch.squeeze(eventNumber)
    jetRawE = torch.squeeze(jetRawE)
    jetRawPt = torch.squeeze(jetRawPt)
    truthJetE = torch.squeeze(truthJetE)
    truthJetPt = torch.squeeze(truthJetPt)
    jetCalE = torch.squeeze(jetCalE)
    r_e_predicted = torch.squeeze(r_e_predicted)
    Cluster_E = torch.squeeze(Cluster_E)
    ClusterPt = torch.squeeze(ClusterPt)
    clusterECalib = torch.squeeze(clusterECalib)
    ClusterENGCALIBTOT = torch.squeeze(ClusterENGCALIBTOT)

    labels_test = torch.squeeze(labels_test)
    nodes_out = torch.squeeze(nodes_out)

    jetCnt = jetCnt.detach().cpu().numpy()
    eventNumber = eventNumber.detach().cpu().numpy()
    jetRawE = jetRawE.detach().cpu().numpy()
    jetRawPt = jetRawPt.detach().cpu().numpy()
    truthJetE = truthJetE.detach().cpu().numpy()
    truthJetPt = truthJetPt.detach().cpu().numpy()
    jetCalE = jetCalE.detach().cpu().numpy()
    r_e_predicted = r_e_predicted.detach().cpu().numpy()
    Cluster_E = Cluster_E.detach().cpu().numpy()
    Cluster_Eta = Cluster_Eta.detach().cpu().numpy()
    Cluster_Phi = Cluster_Phi.detach().cpu().numpy()
    Cluster_Pt = ClusterPt.detach().cpu().numpy()
    clusterECalib = clusterECalib.detach().cpu().numpy()
    ClusterENGCALIBTOT = ClusterENGCALIBTOT.detach().cpu().numpy()

    labels_test = labels_test.detach().cpu().numpy()
    nodes_out = nodes_out.detach().cpu().numpy()


    df_out = pd.DataFrame()
    ## jet counter
    df_out['jetCnt'] = jetCnt
    df_out['eventNumber'] = eventNumber
    ## jet info
    df_out['jetRawE'] = jetRawE
    df_out['jetRawPt'] = jetRawPt
    df_out['truthJetE'] = truthJetE
    df_out['truthJetPt'] = truthJetPt
    df_out['jetCalE'] = jetCalE
    ## cluster info
    df_out['clusterE'] = Cluster_E
    df_out['clusterEta'] = Cluster_Eta
    df_out['clusterPhi'] = Cluster_Phi
    df_out['clusterPt'] = ClusterPt
    df_out['clusterECalib'] = clusterECalib
    df_out['cluster_ENG_CALIB_TOT'] = ClusterENGCALIBTOT
    df_out['r_e_predicted'] = r_e_predicted
    ## labels and scores
    df_out['labels'] = labels_test
    df_out['score'] = nodes_out

    ## now is necesary to undone the normalization
    # load dictionary with means and std
    with open('dict_mean_and_std.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    # norm_variables = ['clusterE', 'clusterEta', 'clusterPhi'] # ['ClusterE','ClusterEta']


    norm_variables = ["clusterE", "clusterEta", 'clusterPhi', "jetRawE" # "nPrimVtx", "avgMu",
                      # "cluster_ENG_FRAC_EM", "cluster_LATERAL", "cluster_LONGITUDINAL",
                      # "cluster_PTD", "cluster_ISOLATION", "zL", "zT", "zRel",'diffEta',
                      # "cluster_CENTER_LAMBDA", "cluster_FIRST_ENG_DENS",
                      # "cluster_SECOND_TIME", "cluster_SIGNIFICANCE",'cluster_CENTER_MAG', "jetRawE"
                     ]
    logged_variables = ["clusterE", "jetRawE",
                        # "cluster_CENTER_LAMBDA", "cluster_FIRST_ENG_DENS",
                        # "cluster_SECOND_TIME", "cluster_SIGNIFICANCE",'cluster_CENTER_MAG',
                       ]
    for field_name in norm_variables:
        df_out[field_name] = undone_Norm(df_out[field_name], loaded_dict[field_name])
        if field_name in logged_variables:
            df_out[field_name] = np.exp(df_out[field_name])
    # df_out['clusterEexp'] = np.exp(df_out['clusterE'])
    df_out['Scores'] = 1 - df_out['score']
    df_out['clusterEDNN'] = df_out['clusterE']  / r_e_predicted



    # grouped_df = df_out.groupby('jetCnt').agg(list).reset_index()

    # for i in grouped_df.columns:
    #     grouped_df[i] = grouped_df[i].apply(lambda x: np.array(x).flatten())

    # output_file = 'output_file.root'
    # # uproot.to_root(output_file, grouped_df, key='GNN')
    # with uproot.recreate(output_file) as f:
    #     f["GNN"] = grouped_df.to_dict(orient="list")

    # df.to_csv(index=False)
    path_to_save = config['data']['path_to_save']
    model_name  = config['data']['model_name']
    # df_out.to_csv(path_to_save+'/ouput_dataframes/out_'+model_name+'.csv')

    root_file_name = path_to_save+'/out_'+model_name+'.root'
    tree_name = 'aTree'

    with uproot3.recreate(root_file_name) as root_file:
        root_file[tree_name] = uproot3.newtree({key: df_out[key].dtype for key in df_out.columns})
        root_file[tree_name].extend(df_out.to_dict(orient='list'))

    # print(df_out)

    return

# Main function call.
if __name__ == '__main__':
    main()
    pass
