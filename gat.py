import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv
import torch_geometric

# GATNet
class GATNet(torch.nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(1, 8, heads=8)
        self.conv2 = GATConv(8 * 8, 8, heads=1)
        self.fc = torch.nn.Linear(8, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = self.fc(x)
        return torch.sigmoid(x).view(-1)
