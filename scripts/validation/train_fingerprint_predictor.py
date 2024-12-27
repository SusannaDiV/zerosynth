import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from math import ceil
from utils import get_atom_stamp, get_shape, get_shape_patches, get_atom_stamp_with_noise
from sklearn.model_selection import train_test_split
from chemprojector.models.encoder import ShapeEncoder, GraphEncoder
from chemprojector.data.common import ProjectionBatch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
import pickle
import argparse

# Without patches augmentation

class FingerprintPredictor(nn.Module):
    def __init__(self, fp_dim=840, encoder_type='shape'):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == 'shape':
            self.encoder = ShapeEncoder(
                patch_size=3,
                d_model=256,
                nhead=8,
                num_layers=6,
                max_seq_length=343
            )
        elif encoder_type == 'graph':
            self.encoder = GraphEncoder(
                num_atom_classes=100,  # From atom_features_simple max value
                num_bond_classes=5,    # From bond_features_simple max value
                dim=256,
                depth=6,
                dim_head=32,
                edge_dim=5,
                heads=8,
                rel_pos_emb=True,
                output_norm=True
            )
        
        self.mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, fp_dim),
            nn.Sigmoid()
        )
    
    def forward(self, batch):
        if self.encoder_type == 'shape':
            # Handle shape input
            if isinstance(batch, torch.Tensor):
                batch = {'shape_patches': batch}
        else:
            # Convert to ProjectionBatch format for graph encoder
            if isinstance(batch, dict):
                # Get batch dimensions
                batch_size = batch['atom_features'].size(0)
                num_atoms = batch['atom_features'].size(1)
                
                # Create edges tensor from edge_index and bond_features
                edges = torch.zeros(batch_size, num_atoms, num_atoms, 5, device=batch['atom_features'].device)
                for b in range(batch_size):
                    edge_idx = batch['edge_index'][b]
                    bond_feat = batch['bond_features'][b]
                    for i in range(edge_idx.size(0)):
                        src, dst = edge_idx[i]
                        edges[b, src.long(), dst.long()] = bond_feat[i][:5]  # Only take first 5 dimensions
                
                # Create the batch dictionary with proper tensor types
                batch = {
                    'atoms': batch['atom_features'].reshape(batch_size, -1, batch['atom_features'].size(-1)).long(),  # [batch_size, num_atoms, feat_dim]
                    'bonds': edges.long(),  # [batch_size, num_atoms, num_atoms, edge_dim]
                    'atom_padding_mask': batch['attention_mask'].reshape(batch_size, -1)  # [batch_size, num_atoms]
                }
        
                # Debugging: Print shapes
                print(f"Atoms shape: {batch['atoms'].shape}")
                print(f"Bonds shape: {batch['bonds'].shape}")
                print(f"Atom padding mask shape: {batch['atom_padding_mask'].shape}")

        # Get encoded representation
        encoded, _ = self.encoder(batch)
        # Use mean pooling over sequence dimension
        pooled = encoded.mean(dim=1)
        # Predict fingerprint
        return self.mlp(pooled)

class ShapeFingerprintDataset(Dataset):
    """Dataset class to handle molecular shape and fingerprint data"""
    def __init__(self, 
                 data,
                 grid_resolution=1.0,
                 max_dist_stamp=3.0,
                 max_dist=10.0,
                 patch_size=3,
                 shape_noise_mu=0.0,
                 shape_noise_sigma=0.0):
        self.data = data
        self.grid_resolution = grid_resolution
        self.max_dist_stamp = max_dist_stamp
        self.max_dist = max_dist
        self.patch_size = patch_size
        self.shape_noise_mu = shape_noise_mu
        self.shape_noise_sigma = shape_noise_sigma
        self.box_size = ceil(2 * max_dist // grid_resolution + 1)
        self.atom_stamp = get_atom_stamp(grid_resolution, max_dist_stamp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Use pre-computed shape instead of computing it again
        shape = item['shape']
        
        # Process shape into patches
        shape_patches = get_shape_patches(shape, self.patch_size)
        shape_patches = shape_patches.reshape(-1, self.patch_size**3)
        
        # Ensure fingerprint is between 0 and 1
        fingerprint = np.clip(item['fingerprint'], 0, 1)

        return {
            'shape': torch.tensor(shape, dtype=torch.float),
            'shape_patches': torch.tensor(shape_patches, dtype=torch.float),
            'fingerprint': torch.from_numpy(fingerprint).float()
        }

def collate_shapes_and_fingerprints(batch):
    """Custom collate function for the dataloader"""
    shapes = []
    shape_patches = []
    fingerprints = []
    
    for item in batch:
        shapes.append(item['shape'])
        shape_patches.append(item['shape_patches'])
        fingerprints.append(item['fingerprint'])

    # Stack all tensors
    shapes = torch.stack(shapes)  # [batch_size, box_size, box_size, box_size]
    shape_patches = torch.stack(shape_patches)  # [batch_size, (box_size // patch_size)**3, patch_size**3]
    fingerprints = torch.stack(fingerprints)  # [batch_size, fp_dim]

    return {
        'shape': shapes,
        'shape_patches': shape_patches,
        'fingerprint': fingerprints
    }

class MoleculeDataset(Dataset):
    """Dataset class to handle molecular graph and fingerprint data"""
    def __init__(self, data):
        # Filter data to only include items with graph_features
        self.data = [item for item in data if 'graph_features' in item]
        if len(self.data) == 0:
            raise ValueError("No items with graph features found in the dataset!")
        print(f"Found {len(self.data)} molecules with graph features")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        graph_features = item['graph_features']
        
        # Ensure all required features are present
        required_features = ['atom_features', 'bond_features', 'edge_index', 'positions', 'attention_mask']
        missing_features = [f for f in required_features if f not in graph_features]
        if missing_features:
            raise ValueError(f"Missing required graph features: {missing_features}")
        
        return {
            'atom_features': torch.tensor(graph_features['atom_features'], dtype=torch.long),
            'bond_features': torch.tensor(graph_features['bond_features'], dtype=torch.long),
            'edge_index': torch.tensor(graph_features['edge_index'], dtype=torch.long),
            'positions': torch.tensor(graph_features['positions'], dtype=torch.float),
            'attention_mask': torch.tensor(graph_features['attention_mask'], dtype=torch.bool),
            'fingerprint': torch.tensor(item['fingerprint'], dtype=torch.float)
        }

def collate_molecules_and_fingerprints(batch):
    """Custom collate function for the molecule dataloader"""
    # Find maximum number of atoms and bonds in the batch
    max_atoms = max(item['atom_features'].size(0) for item in batch)
    max_bonds = max(item['bond_features'].size(0) for item in batch)
    
    # Initialize padded tensors
    batch_size = len(batch)
    atom_features = torch.zeros(batch_size, max_atoms, batch[0]['atom_features'].size(1))
    bond_features = torch.zeros(batch_size, max_bonds, batch[0]['bond_features'].size(1))
    edge_indices = torch.zeros(batch_size, max_bonds, 2)
    positions = torch.zeros(batch_size, max_atoms, 3)
    attention_masks = torch.zeros(batch_size, max_atoms, max_atoms, dtype=torch.bool)
    fingerprints = []
    
    # Fill padded tensors with actual data
    for i, item in enumerate(batch):
        n_atoms = item['atom_features'].size(0)
        n_bonds = item['bond_features'].size(0)
        
        # Copy data
        atom_features[i, :n_atoms] = item['atom_features']
        bond_features[i, :n_bonds] = item['bond_features']
        edge_indices[i, :n_bonds] = item['edge_index'].t()  # Transpose to match expected format
        positions[i, :n_atoms] = item['positions']
        attention_masks[i, :n_atoms, :n_atoms] = item['attention_mask']
        fingerprints.append(item['fingerprint'])
    
    return {
        'atom_features': atom_features.long(),
        'bond_features': bond_features.long(),
        'edge_index': edge_indices.long(),
        'positions': positions.float(),
        'attention_mask': attention_masks,
        'fingerprint': torch.stack(fingerprints)
    }

def train_model(model, train_loader, val_loader, device, num_epochs=50):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch['fingerprint'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch)
                loss = criterion(outputs, batch['fingerprint'])
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_fingerprint_predictor.pt')
            print("Saved new best model!")
        print()

def main():
    # Add argument for encoder type
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='shape', choices=['shape', 'graph'])
    args = parser.parse_args()
    
    # Load dataset
    with open('/itet-stor/sdivita/net_scratch/originale/ChemProjector/data/processed/validation/shape_fingerprint_dataset1.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Split dataset
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Create datasets based on encoder type
    if args.encoder == 'shape':
        train_dataset = ShapeFingerprintDataset(train_data)
        val_dataset = ShapeFingerprintDataset(val_data)
        collate_fn = collate_shapes_and_fingerprints
    else:
        train_dataset = MoleculeDataset(train_data)
        val_dataset = MoleculeDataset(val_data)
        collate_fn = collate_molecules_and_fingerprints
    
    # Create dataloaders with appropriate collate function
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FingerprintPredictor(encoder_type=args.encoder).to(device)
    
    print(f"\nTraining with {args.encoder.upper()} encoder\n")
    
    # Train model
    train_model(model, train_loader, val_loader, device)

if __name__ == "__main__":
    main() 