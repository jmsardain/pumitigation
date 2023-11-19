import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SplineConv, global_mean_pool, DataParallel, EdgeConv, GATConv, GINConv, PNAConv

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
        self.Linear_node1 = torch.nn.Linear(16,32)
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
        self.conv1 = PNAConv(in_channels, out_channels=70, deg=deg, edge_dim=2, towers=2, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv2 = PNAConv(in_channels=70, out_channels=140, deg=deg, edge_dim=2, towers=2, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv3 = PNAConv(in_channels=140, out_channels=280, deg=deg, edge_dim=2, towers=2, post_layers=1,aggregators=aggregators,
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
