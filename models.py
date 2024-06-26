import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SplineConv, global_mean_pool, DataParallel, EdgeConv, GATConv, GINConv, PNAConv, TransformerConv
import torch.nn.functional as F

# Define custom activation functions
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return 2*(x * torch.sigmoid(x))

class TanhPlusOne(nn.Module):
    def __init__(self):
        super(TanhPlusOne, self).__init__()

    def forward(self, x):
        return 2*(torch.tanh(x) + 1)

# Define the model architecture
class PUMitigation(nn.Module):
    def __init__(self, inputsize):
        super(PUMitigation, self).__init__()
        self.flatten = nn.Flatten()
        # self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(in_features=inputsize, out_features=256)
        self.swish1 = Swish()
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.swish2 = Swish()
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.swish3 = Swish()
        self.fc4 = nn.Linear(in_features=64, out_features=8)
        self.swish4 = Swish()
        self.fc5 = nn.Linear(in_features=8, out_features=1)
        self.tanhPlusOne = TanhPlusOne()
        # self.fc6 = nn.Linear(in_features=1, out_features=1)


    def forward(self, x):
        x = self.flatten(x)
        # x = self.dropout(x)
        x = self.swish1(self.fc1(x))
        x = self.swish2(self.fc2(x))
        x = self.swish3(self.fc3(x))
        x = self.swish4(self.fc4(x))
        x = self.tanhPlusOne(self.fc5(x))
        # x = self.fc6(x)
        # x = self.mse_activation(self.fc5(x))
        return x


#### GCNModel reviewed
class GCNModel_2(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GCNModel_2, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        #self.conv3 = GCNConv(hidden_channels, hidden_channels)
        #self.conv4 = GCNConv(hidden_channels, hidden_channels)
        #self.conv5 = GCNConv(hidden_channels, num_classes)

        self.Linear_node1 = torch.nn.Linear(16,32)
        self.Linear_node2 = torch.nn.Linear(32,16)
        #self.Linear_node3 = torch.nn.Linear(32,32)

        self.Linear_node_After1 = torch.nn.Linear(hidden_channels,16)
        self.Linear_node_After2 = torch.nn.Linear(16,16)

        self.Linear_node_After3 = torch.nn.Linear(hidden_channels,16)
        self.Linear_node_After4 = torch.nn.Linear(16,16)

        self.Linear_final1 = torch.nn.Linear(48,64)
        self.Linear_final2 = torch.nn.Linear(64,32)
        self.Linear_final3 = torch.nn.Linear(32,1)

    def forward(self, x, edge_index):
        gg1 = self.Linear_node1(x)
        gg1 = torch.relu(gg1)

        gg2 = self.Linear_node2(gg1)
        gg2 = torch.relu(gg2)

        x1 = self.conv1(x, edge_index)
        x1 = torch.relu(x1)
        x2 = self.conv2(x1, edge_index)
        x2 = torch.relu(x2)
        #x = self.conv3(x, edge_index)
        #x = torch.relu(x)


        x_after1 = self.Linear_node_After1(x1)
        x_after1 = torch.relu(x_after1)
        x_after1 = self.Linear_node_After2(x_after1)
        x_after1 = torch.relu(x_after1)

        x_after2 = self.Linear_node_After3(x2)
        x_after2 = torch.relu(x_after2)
        x_after2 = self.Linear_node_After4(x_after2)
        x_after2 = torch.relu(x_after2)


        xfinal = torch.cat((gg2, x_after1, x_after2), dim=1)
        #xfinal = torch.cat((x, gg3), dim=2)
        xfinal = self.Linear_final1(xfinal)
        xfinal = torch.relu(xfinal)
        xfinal = self.Linear_final2(xfinal)
        xfinal = torch.relu(xfinal)
        xfinal = self.Linear_final3(xfinal)

        x = nn.functional.sigmoid(xfinal)

        return x

#### GATNet reviewed
'''
class GATNet_2(nn.Module):
    def __init__(self, in_channels):
        super(GATNet_2, self).__init__()

        self.conv1 = GATConv(in_channels, 32, heads=8, dropout=0.1)
        #self.conv2 = GCNConv(hidden_channels, num_classes)
        self.conv2 = GATConv(32*8, 32, heads=8, dropout=0.1)
        self.conv3 = GATConv(32*8, 64, heads=12, dropout=0.1)
        #self.conv4 = GCNConv(hidden_channels, hidden_channels)
        #self.conv5 = GCNConv(hidden_channels, num_classes)
        self.Linear_node1 = torch.nn.Linear(in_channels,32)
        self.Linear_node2 = torch.nn.Linear(32,32)
        #self.Linear_node3 = torch.nn.Linear(32,32)

        self.Linear_node_After1 = torch.nn.Linear(32*8,80)
        self.Linear_node_After2 = torch.nn.Linear(80,64)

        self.Linear_node_After3 = torch.nn.Linear(32*8,200)
        self.Linear_node_After4 = torch.nn.Linear(200,64)

        self.Linear_final1 = torch.nn.Linear(64 * 12 + 200 + 80 + 32, 200)
        self.Linear_final2 = torch.nn.Linear(200,64)
        self.Linear_final3 = torch.nn.Linear(64,1)

    def forward(self, x, edge_index):
        gg1 = self.Linear_node1(x)
        gg1 = torch.relu(gg1)

        gg2 = self.Linear_node2(gg1)
        gg2 = torch.relu(gg2)

        x1 = self.conv1(x, edge_index)
        #print(x.size() )
        x1 = torch.relu(x1)
        x2 = self.conv2(x1, edge_index)
        x2 = torch.relu(x2)
        x3 = self.conv3(x2, edge_index)
        x3 = torch.relu(x3)

        x_after1 = self.Linear_node_After1(x1)
        x_after1 = torch.relu(x_after1)
        x_after1 = self.Linear_node_After2(x_after1)
        x_after1 = torch.relu(x_after1)

        x_after2 = self.Linear_node_After3(x2)
        x_after2 = torch.relu(x_after2)
        x_after2 = self.Linear_node_After4(x_after2)
        x_after2 = torch.relu(x_after2)


        xfinal = torch.cat(( gg2, x3, x_after1, x_after2), dim=1)
        #xfinal = torch.cat((x, gg3), dim=2)
        xfinal = self.Linear_final1(xfinal)
        xfinal = torch.relu(xfinal)
        xfinal = self.Linear_final2(xfinal)
        xfinal = torch.relu(xfinal)
        xfinal = self.Linear_final3(xfinal)

        #x = nn.functional.sigmoid(x)
        x = nn.functional.sigmoid(xfinal)

        return x
'''
#### GATNet original
class GATNet(nn.Module):
    def __init__(self, in_channels):
        super(GATNet, self).__init__()

        self.conv1 = GATConv(in_channels, 16, heads=8, dropout=0.1)
        self.conv2 = GATConv(16*8, 32, heads=8, dropout=0.1)

        self.Linear_node1 = torch.nn.Linear(16,32)
        self.Linear_node2 = torch.nn.Linear(32,32)

        self.Linear_node_After1 = torch.nn.Linear(128,64)
        self.Linear_node_After2 = torch.nn.Linear(64,64)

        self.Linear_node_After3 = torch.nn.Linear(256,128)
        self.Linear_node_After4 = torch.nn.Linear(128,64)

        self.Linear_final1 = torch.nn.Linear(160,128)
        self.Linear_final2 = torch.nn.Linear(128,64)
        self.Linear_final3 = torch.nn.Linear(64,1)

    def forward(self, x, edge_index):

        gg1 = self.Linear_node1(x)
        gg1 = torch.relu(gg1)

        gg2 = self.Linear_node2(gg1)
        gg2 = torch.relu(gg2)

        x1 = self.conv1(x, edge_index)
        x1 = torch.relu(x1)
        x2 = self.conv2(x1, edge_index)
        x2 = torch.relu(x2)

        x_after1 = self.Linear_node_After1(x1)
        x_after1 = torch.relu(x_after1)
        x_after1 = self.Linear_node_After2(x_after1)
        x_after1 = torch.relu(x_after1)

        x_after2 = self.Linear_node_After3(x2)
        x_after2 = torch.relu(x_after2)
        x_after2 = self.Linear_node_After4(x_after2)

        x_after2 = torch.relu(x_after2)


        xfinal = torch.cat((gg2, x_after1, x_after2), dim=1)
        xfinal = self.Linear_final1(xfinal)
        xfinal = torch.relu(xfinal)
        xfinal = self.Linear_final2(xfinal)
        xfinal = torch.relu(xfinal)
        xfinal = self.Linear_final3(xfinal)

        x = nn.functional.sigmoid(xfinal)

        return x


class GATNet_2(nn.Module):
    def __init__(self, in_channels):
        super(GATNet_2, self).__init__()

        self.conv1 = GATConv(in_channels, 32, heads=8, dropout=0.1)
        self.conv2 = GATConv(32*8, 32, heads=8, dropout=0.1)
        self.conv3 = GATConv(32*8, 64, heads=12, dropout=0.1)

        self.Linear_node1 = torch.nn.Linear(in_channels,32)
        self.Linear_node2 = torch.nn.Linear(32,32)

        self.Linear_node_After1 = torch.nn.Linear(32*8,80)
        #self.Linear_node_After2 = torch.nn.Linear(64,64)

        self.Linear_node_After3 = torch.nn.Linear(32*8,200)
        #self.Linear_node_After4 = torch.nn.Linear(128,64)

        self.Linear_final1 = torch.nn.Linear(64*12+200 + 80 + 32,200)
        self.Linear_final2 = torch.nn.Linear(200,64)
        self.Linear_final3 = torch.nn.Linear(64,1)

    def forward(self, x, edge_index):
        gg1 = self.Linear_node1(x)
        gg1 = torch.relu(gg1)

        gg2 = self.Linear_node2(gg1)
        gg2 = torch.relu(gg2)

        x1 = self.conv1(x, edge_index)
        x1 = torch.relu(x1)
        x2 = self.conv2(x1, edge_index)
        x2 = torch.relu(x2)
        x3 = self.conv3(x2, edge_index)
        x3 = torch.relu(x3)

        x_after1 = self.Linear_node_After1(x1)
        x_after1 = torch.relu(x_after1)
        #x_after1 = self.Linear_node_After2(x_after1)
        #x_after1 = torch.relu(x_after1)

        x_after2 = self.Linear_node_After3(x2)
        x_after2 = torch.relu(x_after2)
        #x_after2 = self.Linear_node_After4(x_after2)
        #x_after2 = torch.relu(x_after2)


        xfinal = torch.cat(( gg2, x3, x_after1, x_after2), dim=1)
        xfinal = self.Linear_final1(xfinal)
        xfinal = torch.relu(xfinal)
        xfinal = self.Linear_final2(xfinal)
        xfinal = torch.relu(xfinal)
        xfinal = self.Linear_final3(xfinal)

        x = nn.functional.sigmoid(xfinal)

        return x


class GATNet_JMS(nn.Module):
    def __init__(self, in_channels, weight_decay):
        super(GATNet_JMS, self).__init__()

        self.conv1 = GATConv(in_channels, 32, heads=8, dropout=0.1)
        self.conv2 = GATConv(32*8, 32, heads=8, dropout=0.1)
        self.conv3 = GATConv(32*8, 64, heads=12, dropout=0.1)

        self.Linear_node1 = torch.nn.Linear(in_channels,32)
        self.Linear_node2 = torch.nn.Linear(32,32)

        self.Linear_node_After1 = torch.nn.Linear(32*8,80)
        #self.Linear_node_After2 = torch.nn.Linear(64,64)

        self.Linear_node_After3 = torch.nn.Linear(32*8,200)
        #self.Linear_node_After4 = torch.nn.Linear(128,64)

        self.Linear_final1 = torch.nn.Linear(64*12+200 + 80 + 32,200)
        self.Linear_final2 = torch.nn.Linear(200,64)
        self.Linear_final3 = torch.nn.Linear(64,1)

        self.weight_decay = weight_decay
    def forward(self, x, edge_index):
        gg1 = self.Linear_node1(x)
        gg1 = torch.relu(gg1)

        gg2 = self.Linear_node2(gg1)
        gg2 = torch.relu(gg2)

        x1 = self.conv1(x, edge_index)
        x1 = torch.relu(x1)
        x2 = self.conv2(x1, edge_index)
        x2 = torch.relu(x2)
        x3 = self.conv3(x2, edge_index)
        x3 = torch.relu(x3)

        x_after1 = self.Linear_node_After1(x1)
        x_after1 = torch.relu(x_after1)
        x_after2 = self.Linear_node_After3(x2)
        x_after2 = torch.relu(x_after2)

        xfinal = torch.cat(( gg2, x3, x_after1, x_after2), dim=1)
        xfinal = self.Linear_final1(xfinal)
        xfinal = torch.relu(xfinal)
        xfinal = self.Linear_final2(xfinal)
        xfinal = torch.relu(xfinal)
        xfinal = self.Linear_final3(xfinal)

        l2_reg = torch.tensor(0.0, device=x.device)
        for param in self.parameters():
            l2_reg += torch.norm(param)

        x = torch.sigmoid(xfinal)

        return x, self.weight_decay * l2_reg


######### --------------
class PNAConv_OnlyNodes(nn.Module):
    def __init__(self, in_channels):
        super(PNAConv_OnlyNodes, self).__init__()
        aggregators = ['sum','mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation',"linear",'inverse_linear']
        #'''
        self.conv1 = PNAConv(in_channels, out_channels=32, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv2 = PNAConv(in_channels=32, out_channels=64, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv3 = PNAConv(in_channels=64, out_channels=128, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        '''
        self.conv1 = PNA(in_channels=-1, hidden_channels = 32 , num_layers=1 , out_channels=64, aggregators=aggregators,
                                            scalers = scalers)
        self.conv2 = PNA(in_channels=32, out_channels=64, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv3 = PNA(in_channels=64, out_channels=128, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        '''
        self.Linear_node1 = torch.nn.Linear(in_channels,32)
        #self.Linear_node2 = torch.nn.Linear(32,32)
        #self.Linear_node3 = torch.nn.Linear(32,32)

        self.Linear_node_After1 = torch.nn.Linear(32,64)
        #self.Linear_node_After2 = torch.nn.Linear(64,64)

        self.Linear_node_After3 = torch.nn.Linear(64,128)
        #self.Linear_node_After4 = torch.nn.Linear(128,64)

        #self.Linear_final1 = torch.nn.Linear(128 + 32*8 + 32,160)
        self.Linear_final1 = torch.nn.Linear(32 + 64 + 128 + 128,300)
        self.Linear_final2 = torch.nn.Linear(300,124)
        self.Linear_final3 = torch.nn.Linear(124,1)

    def forward(self, x, edge_index):

        ## made some changes by mistake, test again and check layers sizes
        gg1 = self.Linear_node1(x)
        gg1 = torch.relu(gg1)

        #gg2 = self.Linear_node2(gg1)
        #gg2 = torch.relu(gg2)

        x1 = self.conv1(x, edge_index)
        #x1 = torch.relu(x1)
        x2 = self.conv2(x1, edge_index)
        #x2 = torch.relu(x2)
        x3 = self.conv3(x2, edge_index)
        #x3 = torch.relu(x3)

        x_after1 = self.Linear_node_After1(x1)
        x_after1 = torch.relu(x_after1)
        #x_after1 = self.Linear_node_After2(x_after1)
        #x_after1 = torch.relu(x_after1)

        x_after2 = self.Linear_node_After3(x2)
        x_after2 = torch.relu(x_after2)
        #x_after2 = self.Linear_node_After4(x_after2)
        #x_after2 = torch.relu(x_after2)


        #xfinal = torch.cat(( gg2, x3, x_after1, x_after2), dim=1)
        xfinal = torch.cat(( gg1, x3, x_after1, x_after2), dim=1)
        #xfinal = torch.cat((x, gg3), dim=2)
        xfinal = self.Linear_final1(xfinal)
        xfinal = torch.relu(xfinal)
        xfinal = self.Linear_final2(xfinal)
        xfinal = torch.relu(xfinal)
        xfinal = self.Linear_final3(xfinal)

        #x = nn.functional.sigmoid(x)
        x = nn.functional.sigmoid(xfinal)

        return x

#### models including edge features
class PNAConv_EdgeAttrib(nn.Module):
    def __init__(self, in_channels, deg):
        super(PNAConv_EdgeAttrib, self).__init__()
        aggregators = ['sum','mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation',"linear",'inverse_linear']
        #'''
        self.conv1 = PNAConv(in_channels, out_channels=70, deg=deg, edge_dim=22, towers=2, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv2 = PNAConv(in_channels=70, out_channels=140, deg=deg, edge_dim=22, towers=2, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv3 = PNAConv(in_channels=140, out_channels=280, deg=deg, edge_dim=22, towers=2, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        '''
        self.conv1 = PNA(in_channels=-1, hidden_channels = 32 , num_layers=1 , out_channels=64, aggregators=aggregators,
                                            scalers = scalers)
        self.conv2 = PNA(in_channels=32, out_channels=64, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv3 = PNA(in_channels=64, out_channels=128, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        '''
        #self.Linear_node1 = torch.nn.Linear(16,32)
        #self.Linear_node2 = torch.nn.Linear(32,32)
        #self.Linear_node3 = torch.nn.Linear(32,32)

        self.Linear_node_After1 = torch.nn.Linear(32,64)
        #self.Linear_node_After2 = torch.nn.Linear(64,64)

        self.Linear_node_After3 = torch.nn.Linear(64,128)
        #self.Linear_node_After4 = torch.nn.Linear(128,64)

        #self.Linear_final1 = torch.nn.Linear(0 + 40 + 100 + 230,370)

        #self.Linear_final1 = torch.nn.Linear(0 + 64 + 128 + 256,370)
        #self.Linear_final2 = torch.nn.Linear(370,150)
        self.Linear_final1 = torch.nn.Linear(0 + 70 + 140 + 280,420)
        self.Linear_final2 = torch.nn.Linear(420,180)
        self.Linear_final3 = torch.nn.Linear(180,60)
        self.Linear_final4 = torch.nn.Linear(60,1)

        self.Drop = nn.Dropout( 0.15 )
    def forward(self, x, edge_index, edge_attr):

        ## made some changes by mistake, test again and check layers sizes
        #gg1 = self.Linear_node1(x)
        #gg1 = torch.relu(gg1)

        #gg2 = self.Linear_node2(gg1)
        #gg2 = torch.relu(gg2)

        x1 = self.conv1(x, edge_index, edge_attr)
        #x1 = torch.relu(x1)
        x2 = self.conv2(x1, edge_index, edge_attr)
        #x2 = torch.relu(x2)
        x3 = self.conv3(x2, edge_index, edge_attr)
        #x3 = torch.relu(x3)

        #x_after1 = self.Linear_node_After1(x1)
        #x_after1 = torch.relu(x_after1)
        #x_after1 = self.Linear_node_After2(x_after1)
        #x_after1 = torch.relu(x_after1)

        #x_after2 = self.Linear_node_After3(x2)
        #x_after2 = torch.relu(x_after2)
        #x_after2 = self.Linear_node_After4(x_after2)
        #x_after2 = torch.relu(x_after2)


        #xfinal = torch.cat(( gg2, x3, x_after1, x_after2), dim=1)
        #xfinal = torch.cat(( gg1, x3, x_after1, x_after2), dim=1)
        #xfinal = torch.cat(( x3, x_after1, x_after2), dim=1)
        xfinal = torch.cat(( x3, x2, x1), dim=1)
        #xfinal = torch.cat((x, gg3), dim=2)
        xfinal = self.Linear_final1(xfinal)
        x = self.Drop(x)
        xfinal = torch.relu(xfinal)
        xfinal = self.Linear_final2(xfinal)
        x = self.Drop(x)
        xfinal = torch.relu(xfinal)
        xfinal = self.Linear_final3(xfinal)
        x = self.Drop(x)
        xfinal = torch.relu(xfinal)
        xfinal = self.Linear_final4(xfinal)

        #x = nn.functional.sigmoid(x)
        x = nn.functional.sigmoid(xfinal)

        return x


class EdgeGinNet(torch.nn.Module):
    def __init__(self, in_channels):
        super(EdgeGinNet, self).__init__()
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(in_channels*2, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(),
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),aggr='add', dropout=0.1)
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(64, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(),
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),aggr='add', dropout=0.1)
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(64,64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),aggr='add', dropout=0.1)
        self.conv4 = EdgeConv(nn.Sequential(nn.Linear(128, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),aggr='add', dropout=0.1)
        self.conv5 = EdgeConv(nn.Sequential(nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),aggr='add', dropout=0.1)
        self.conv6 = EdgeConv(nn.Sequential(nn.Linear(256, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),aggr='add', dropout=0.1)

        self.seq1 = nn.Sequential(nn.Linear(448, 384),
                                nn.BatchNorm1d(num_features=384),
                                nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(384, 256),
                                  nn.ReLU())
        #self.lin1 = torch.nn.Linear(128, 128)
        #self.lin2 = torch.nn.Linear(448, 64)
        self.lin = nn.Linear(256, 1)

    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        x6 = self.conv6(x5, edge_index)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x = self.seq1(x)
        # x = global_mean_pool(x, batch)
        x = self.seq2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        #print(x.shape)
        return F.sigmoid(x)


class GATNet_simple_withMLP(nn.Module):
    def __init__(self, in_channels):
        super(GATNet_simple_withMLP, self).__init__()

        self.conv1 = GATConv(in_channels, 32, heads=8, dropout=0.1)
        self.conv2 = GATConv(32 * 8, 32, heads=8, dropout=0.1)
        self.conv3 = GATConv(32 * 8, 64, heads=12, dropout=0.1)
        self.linear_final = nn.Linear(32*8 + 32*8 + 64*12, 1)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 1, 128),  # add first output to input
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  #  output between 0 and 1
        )

    def forward(self, x, edge_index, JetRawPt):

        x1 = self.conv1(x, edge_index)
        x1 = torch.relu(x1)
        x2 = self.conv2(x1, edge_index)
        x2 = torch.relu(x2)
        x3 = self.conv3(x2, edge_index)
        x3 = torch.relu(x3)
        x_concat = torch.cat([x1, x2, x3], dim=1)
        x_final = self.linear_final(x_concat)
        output1 = torch.sigmoid(x_final)

        # add classification output to input features
        x_combined = torch.cat([x, output1], dim=1)

        # use the new features in MLP
        output2 = self.mlp(x_combined)
        return output1, output2



class GATNet_simple(nn.Module):
    def __init__(self, in_channels):
        super(GATNet_simple, self).__init__()

        self.conv1 = GATConv(in_channels, 32, heads=8, dropout=0.1)
        self.conv2 = GATConv(32 * 8, 32, heads=8, dropout=0.1)
        self.conv3 = GATConv(32 * 8, 64, heads=12, dropout=0.1)
        self.linear_final = nn.Linear(32*8 + 32*8 + 64*12, 1)

    def forward(self, x, edge_index, JetRawPt):

        x1 = self.conv1(x, edge_index)
        x1 = torch.relu(x1)
        x2 = self.conv2(x1, edge_index)
        x2 = torch.relu(x2)
        x3 = self.conv3(x2, edge_index)
        x3 = torch.relu(x3)
        x_concat = torch.cat([x1, x2, x3], dim=1)
        x_final = self.linear_final(x_concat)
        return torch.sigmoid(x_final)

class GATNet_simple_withEdges(nn.Module):
    def __init__(self, in_channels):
        super(GATNet_simple_withEdges, self).__init__()

        self.conv1 = GATConv(in_channels, 32, heads=8, dropout=0.1, edge_dim=22)
        self.conv2 = GATConv(32 * 8, 32, heads=8, dropout=0.1, edge_dim=22)
        self.conv3 = GATConv(32 * 8, 64, heads=12, dropout=0.1, edge_dim=22)

        self.linear_final = nn.Linear(32*8 + 32*8 + 64*12, 1)

    def forward(self, x, edge_index, edge_attr):

        x1 = self.conv1(x, edge_index, edge_attr)
        x1 = torch.relu(x1)
        x2 = self.conv2(x1, edge_index, edge_attr)
        x2 = torch.relu(x2)
        x3 = self.conv3(x2, edge_index, edge_attr)
        x3 = torch.relu(x3)

        x_concat = torch.cat([x1, x2, x3], dim=1)

        x_final = self.linear_final(x_concat)

        return torch.sigmoid(x_final)


class GraphTransformer(nn.Module):
    def __init__(self, in_channels):
        super(GraphTransformer, self).__init__()
        self.conv1 = TransformerConv(in_channels, 32, heads=8, edge_dim=22)
        self.conv2 = TransformerConv(32 * 8, 64, heads=8, edge_dim=22)
        self.linear = nn.Linear(32*8 + 64*8, 1)

    def forward(self, x, edge_index, edge_attr):
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x1 = self.conv1(x, edge_index, edge_attr)
        x1 = torch.relu(x1)
        x2 = self.conv2(x1, edge_index, edge_attr)
        x2 = torch.relu(x2)
        x_concat = torch.cat([x1, x2], dim=1)

        x_final = self.linear(x_concat)
        return torch.sigmoid(x_final)


class GATNet_with_pT_attention(nn.Module):
    def __init__(self, in_channels):
        super(GATNet_with_pT_attention, self).__init__()

        self.conv1 = GATConv(in_channels, 32, heads=8, dropout=0.1)
        self.conv2 = GATConv(32 * 8, 32, heads=8, dropout=0.1)
        self.conv3 = GATConv(32 * 8, 64, heads=12, dropout=0.1)

        # Attention mechanism for pT values
        self.attention_pT = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1280),
            nn.Sigmoid()
        )

        self.linear_final = nn.Linear(32*8 + 32*8 + 64*12, 1)

    def forward(self, x, edge_index, JetRawPt):

        x1 = self.conv1(x, edge_index)
        x1 = torch.relu(x1)
        x2 = self.conv2(x1, edge_index)
        x2 = torch.relu(x2)
        x3 = self.conv3(x2, edge_index)
        x3 = torch.relu(x3)

        x_concat = torch.cat([x1, x2, x3], dim=1)

        pT_weights = self.attention_pT(JetRawPt.unsqueeze(1))

        pT_weights = pT_weights.unsqueeze(1)
        x_weighted = x_concat * pT_weights

        x_weighted_sum = torch.sum(x_weighted, dim=1)

        x_final = self.linear_final(x_weighted_sum)

        return torch.sigmoid(x_final)
