import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from math import ceil
from utils import get_atom_stamp, get_shape, get_shape_patches, get_atom_stamp_with_noise
from sklearn.model_selection import train_test_split
from chemprojector.models.encoder import ShapeEncoder
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
import pickle

# Without patches augmentation

class FingerprintPredictor(nn.Module):
    def __init__(self, fp_dim=840):
        super().__init__()
        self.encoder = ShapeEncoder(
            patch_size=3,
            d_model=256,
            nhead=8,
            num_layers=6,
            max_seq_length=343  # (21/3)^3 = 343 patches max
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, fp_dim),
            nn.Sigmoid()
        )
    
    def forward(self, batch):
        # If batch is a tensor, wrap it in a dict
        if isinstance(batch, torch.Tensor):
            batch = {'shape_patches': batch}
        
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

def create_unique_fingerprint_split(data, test_size=0.2, random_state=42):
    """Split data ensuring no fingerprint overlap between train and validation sets"""
    # Create dictionary mapping fingerprints to all their associated records
    fp_to_records = {}
    for idx, item in enumerate(data):
        # Convert fingerprint array to tuple for hashability
        fp_tuple = tuple(item['fingerprint'].flatten())
        if fp_tuple not in fp_to_records:
            fp_to_records[fp_tuple] = []
        fp_to_records[fp_tuple].append(idx)
    
    # Get unique fingerprints and split them
    unique_fps = list(fp_to_records.keys())
    np.random.seed(random_state)
    np.random.shuffle(unique_fps)
    
    split_idx = int(len(unique_fps) * (1 - test_size))
    train_fps = unique_fps[:split_idx]
    val_fps = unique_fps[split_idx:]
    
    # Create train and validation datasets
    train_indices = []
    val_indices = []
    
    for fp in train_fps:
        train_indices.extend(fp_to_records[fp])
    for fp in val_fps:
        val_indices.extend(fp_to_records[fp])
    
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    
    print(f"Split dataset into {len(train_data)} training and {len(val_data)} validation samples")
    print(f"Using {len(train_fps)} unique fingerprints for training and {len(val_fps)} for validation")
    
    return train_data, val_data

def main():
    # Load dataset
    with open('/itet-stor/sdivita/net_scratch/originale/ChemProjector/data/processed/validation/dataset2.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Split dataset ensuring unique fingerprints
    train_data, val_data = create_unique_fingerprint_split(data, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = ShapeFingerprintDataset(train_data)
    val_dataset = ShapeFingerprintDataset(val_data)
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_shapes_and_fingerprints)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_shapes_and_fingerprints)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FingerprintPredictor().to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, device)

if __name__ == "__main__":
    main() 