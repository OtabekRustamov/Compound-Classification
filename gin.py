import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv
import torch_geometric

# GINConvNet
class GINConvNet(torch.nn.Module):
    def __init__(self):
        super(GINConvNet, self).__init__()
        self.conv1 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(1, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8)
        ))
        self.conv2 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8)
        ))
        self.fc = torch.nn.Linear(8, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = self.fc(x)
        return torch.sigmoid(x).view(-1)