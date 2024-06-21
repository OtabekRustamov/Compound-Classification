from torch_geometric.loader import DataLoader
from gat import GATNet
from gat_gcn import GAT_GCN
from gin import GINConvNet
import torch
import numpy as np
import pandas as pd
from data_preprocessing import smiles_to_graph
from torch_geometric.loader import DataLoader

# Load the dataset
df = pd.read_csv('cmpd.csv')

# Apply the conversion and filter out None values
df['graph'] = df.apply(lambda row: smiles_to_graph(row['smiles'], row['activity'] == 'active'), axis=1)
df = df.dropna(subset=['graph'])

# Split data
train_data = df[df.group == 'train']
test_data = df[df.group == 'test']


train_loader = DataLoader(train_data['graph'].tolist(), batch_size=32, shuffle=True)
test_loader = DataLoader(test_data['graph'].tolist(), batch_size=32)

# Train function
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)  # Move to GPU if available
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Test function
def test(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)  # Move to GPU if available
            output = model(data)
            loss = criterion(output, data.y)
            total_loss += loss.item()
    return total_loss / len(loader)

# Initialize and train models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_classes = [GATNet, GAT_GCN, GINConvNet]
model_names = ["GATNet", "GAT_GCN", "GINConvNet"]

for model_class, model_name in zip(model_classes, model_names):
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()
    print(f'Training {model_name}')
    for epoch in range(50):
        train_loss = train(model, train_loader, optimizer, criterion)
        test_loss = test(model, test_loader, criterion)
        print(f'{model_name} - Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')