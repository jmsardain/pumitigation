import torch
from torch_geometric.data import Data


def makeGraph(feature, label):

    ## keep this in mind
    # column_names = ['clusterE', 'clusterEta', 'cluster_CENTER_LAMBDA', 'cluster_CENTER_MAG',
    #             'cluster_ENG_FRAC_EM', 'cluster_FIRST_ENG_DENS', 'cluster_LATERAL', 'cluster_LONGITUDINAL',
    #             'cluster_PTD', 'cluster_time', 'cluster_ISOLATION', 'cluster_SECOND_TIME', 'cluster_SIGNIFICANCE',
    #             'nPrimVtx', 'avgMu']
    graph_list = []
    for i in range(len(feature)):
        num_nodes = len(feature[i])
        edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(i+1, num_nodes)], dtype=torch.long).t().contiguous()

        vec = []

        vec.append(np.array([feature[i][0],  feature[i][1],  feature[i][2],  feature[i][3],  feature[i][4],
                             feature[i][5],  feature[i][6],  feature[i][7],  feature[i][8],  feature[i][9],
                             feature[i][10], feature[i][11], feature[i][12], feature[i][13], feature[i][14],
                             feature[i][15]]).T)


        vec = np.array(vec)
        vec = np.squeeze(vec)

        x = torch.tensor(vec, dtype=torch.float)
        w = (np.array(label[i]) > 0.5)*9 + 1


        #graph = Data(x=x, edge_index=edge_index, y=labels[i])
        graph = Data(x=x, edge_index=edge_index, y=torch.tensor(label[i], dtype=torch.float), weights=torch.tensor(w, dtype=torch.float) )
        graph_list.append(graph)

    return graph_list
