import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data,DataLoader
from torch_geometric.nn import GCNConv
from torch_scatter import  scatter_max,scatter_mean,scatter_sum
from torch_geometric.nn import GATConv, MLP, GatedGraphConv
from torch import nn, einsum

import config as cf

INPUT_SIZE = cf.image_size[1] #10
OUTPUT_SIZE = cf.num_classes
class Net_GCN(torch.nn.Module):
    def __init__(self):
        super(Net_GCN, self).__init__()
        self.conv1 = GCNConv(INPUT_SIZE, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 64)
        self.linear1 = torch.nn.Linear(64,256)
        self.linear2 = torch.nn.Linear(256,OUTPUT_SIZE)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x, _ = scatter_max(x, data.batch, dim=0)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return self.sigmoid(x)
 
class Net_GAT(torch.nn.Module):
    def __init__(self):
        super(Net_GAT, self).__init__()
        self.conv1 = GATConv(INPUT_SIZE, 16, 2)
        self.conv2 = GATConv(16*2, 32, 1)
        #self.conv3 = GATConv(64*4, 128, 1)
        self.linear1 = torch.nn.Linear(32,64)
        #self.linear2 = torch.nn.Linear(256,256)
        self.linear2 = torch.nn.Linear(64,OUTPUT_SIZE)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = self.conv3(x, edge_index)
        #x = F.relu(x)

        #x, _= scatter_max(x, data.batch, dim=0)
        x= scatter_sum(x, data.batch, dim=0)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        #x = F.relu(x)
        #x = self.linear3(x)
        return self.sigmoid(x)
    

class Net_GatedGNN(torch.nn.Module):
    def __init__(self):
        super(Net_GatedGNN, self).__init__()
        self.conv1 = GatedGraphConv(INPUT_SIZE, 32)
        self.conv2 = GatedGraphConv(32, 32)
        #self.conv3 = GATConv(64*4, 128, 1)
        self.linear1 = torch.nn.Linear(32,64)
        #self.linear2 = torch.nn.Linear(256,256)
        self.linear2 = torch.nn.Linear(64,OUTPUT_SIZE)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = self.conv3(x, edge_index)
        #x = F.relu(x)

        #x, _= scatter_max(x, data.batch, dim=0)
        x= scatter_sum(x, data.batch, dim=0)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        #x = F.relu(x)
        #x = self.linear3(x)
        return self.sigmoid(x)
    
    
class Net_MLP(nn.Module):
    def __init__(self):
        super(Net_MLP, self).__init__()

        self.linear1 = nn.Linear(cf.image_size[1]*cf.image_size[0], 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, OUTPUT_SIZE)
        self.linear_h = nn.Linear(64, 64)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, data):
        x= data.x.view(-1,cf.image_size[1]*cf.image_size[0])
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear_h(x)
        x = self.relu(x)
        x = self.linear_h(x)
        x = self.relu(x)
        x = self.linear2(x)
        return self.sigmoid(x)