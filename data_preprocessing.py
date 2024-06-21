import pandas as pd
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import numpy as np

import pandas as pd
import torch
from torch_geometric.data import Data
import torch_geometric
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import numpy as np




# Convert SMILES to molecular graphs
def smiles_to_graph(smiles, activity):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Get adjacency matrix
    adj = GetAdjacencyMatrix(mol)

    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom.GetAtomicNum())

    # Convert to torch tensor
    edge_index = np.array(adj.nonzero())
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    x = torch.tensor(atom_features, dtype=torch.float).unsqueeze(-1)

    # Convert activity to tensor
    y = torch.tensor([activity], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)





