import torch
from torch_geometric.data import Data


def makeGraph():
    graph_list = []
    for i in range(len(clusterE)):
        num_nodes = len(clusterE[i])
        edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(i+1, num_nodes)], dtype=torch.long).t().contiguous()
        #print(edge_index)
        vec = []

        vec.append(np.array([Dictionary_data["0"][i], Dictionary_data["1"][i], Dictionary_data["2"][i], Dictionary_data["3"][i], Dictionary_data["4"][i],
                             Dictionary_data["5"][i], Dictionary_data["6"][i], Dictionary_data["7"][i], Dictionary_data["8"][i], Dictionary_data["9"][i],
                             Dictionary_data["10"][i], Dictionary_data["11"][i], Dictionary_data["12"][i], Dictionary_data["13"][i], Dictionary_data["14"][i],
                             Dictionary_data["15"][i]]).T)


        vec = np.array(vec)
        vec = np.squeeze(vec)

        x = torch.tensor(vec, dtype=torch.float)
        w=(np.array(labels[i]) > 0.5)*9 + 1


        #graph = Data(x=x, edge_index=edge_index, y=labels[i])
        graph = Data(x=x, edge_index=edge_index, y=torch.tensor(labels[i], dtype=torch.float), weights=torch.tensor(w, dtype=torch.float) )
        graph_list.append(graph)
    return graph_list
