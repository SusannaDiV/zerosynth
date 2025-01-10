Run SBDD tasks on:
1. Shape encoder as is + Morgan fingerprint
2. Shape encoder as is + Morgan fingerprint + E3P fingerprint
3. Shape encoder with pharmacofore features 
    Compute the pharmacofore feature of the whole molecule
    
    import torch
from torch import nn
import numpy as np
from typing import List, Tuple
from chemprojector.data.common import ProjectionBatch
from chemprojector.chem.gaussian_ph4 import GaussianPH4Generator, FEATURE_TYPES

class ShapeEncoderWithPH4(BaseEncoder):
    def __init__(
        self,
        patch_size: int = 3,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        max_seq_length: int = 3000,
        ph4_feature_types: List[str] = FEATURE_TYPES,  # ['ARO', 'HBD', 'HBA', 'POS', 'NEG', 'HYD']
        ph4_sigma: float = 1.0,
        grid_resolution: float = 0.5,  # Å
        dropout: float = 0.1
    ):
        super().__init__()
        self._dim = d_model
        self._patch_size = patch_size
        self.grid_resolution = grid_resolution
        
        # Shape processing
        self._patch_ffn = nn.Sequential(
            nn.Linear(patch_size**3, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Pharmacophore processing
        self.num_ph4_types = len(ph4_feature_types)
        self._ph4_embedding = nn.Embedding(self.num_ph4_types + 1, d_model, padding_idx=0)  # +1 for empty patches
        
        # Fusion module
        self._fusion_ffn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Transformer components
        self._pos_embed = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        self._embed_dropout = nn.Dropout(dropout)
        self._transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_layers
        )
        self._norm = nn.LayerNorm(d_model)
        
        # Initialize pharmacophore generator
        self.ph4_generator = GaussianPH4Generator(
            sigma=ph4_sigma,
            feature_types=ph4_feature_types
        )
'''
    def get_patch_centers(self, grid_origin: torch.Tensor, num_patches: int) -> torch.Tensor:
        """Compute centers of all patches"""
        patches_per_dim = int(np.cbrt(num_patches))
        patch_size_ang = self._patch_size * self.grid_resolution
        
        centers = []
        for i in range(patches_per_dim):
            for j in range(patches_per_dim):
                for k in range(patches_per_dim):
                    center = grid_origin + torch.tensor([
                        (i + 0.5) * patch_size_ang,
                        (j + 0.5) * patch_size_ang,
                        (k + 0.5) * patch_size_ang
                    ], device=grid_origin.device)
                    centers.append(center)
                    
        return torch.stack(centers)  # [num_patches, 3]

    def assign_features_to_patches(
        self,
        ph4_coords: torch.Tensor,  # [N_features, 3]
        ph4_types: torch.Tensor,   # [N_features]
        patch_centers: torch.Tensor,  # [num_patches, 3]
        patch_size: float
    ) -> torch.Tensor:
        """Assign pharmacophore features to patches based on spatial proximity"""
        num_patches = patch_centers.size(0)
        half_size = patch_size / 2
        
        # Initialize feature counts per patch
        patch_features = torch.zeros(
            (num_patches, self.num_ph4_types),
            device=ph4_coords.device
        )
        
        # For each feature
        for feat_idx in range(len(ph4_coords)):
            feat_coord = ph4_coords[feat_idx]
            feat_type = ph4_types[feat_idx]
            
            # Find which patches this feature belongs to
            distances = torch.norm(patch_centers - feat_coord.unsqueeze(0), dim=1)
            in_patch = distances < half_size
            
            # Add feature to relevant patches
            patch_features[in_patch, feat_type] += 1
            
        return patch_features  # [num_patches, num_ph4_types]
'''
    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, batch: ProjectionBatch) -> tuple[torch.Tensor, torch.Tensor]:
        if "shape_patches" not in batch or "mol" not in batch or "grid_origin" not in batch:
            raise ValueError("shape_patches, mol and grid_origin must be in batch")
            
        shape_patches = batch["shape_patches"]  # [batch_size, num_patches, patch_size^3]
        bz, num_patches, _ = shape_patches.size()
        
        # Process shape patches
        x_shape = self._patch_ffn(shape_patches)  # [batch_size, num_patches, d_model]
        
        # Generate pharmacophore features for whole molecule once
        mol_ph4_features = self.ph4_generator(batch["mol"])  # Returns coordinates and types
        mol_ph4_coords = mol_ph4_features[:, :3]  # [N_features, 3]
        mol_ph4_types = mol_ph4_features[:, 3].long()  # [N_features]
        
        # Get patch centers
        patch_centers = self.get_patch_centers(
            batch["grid_origin"], 
            num_patches
        )  # [num_patches, 3]
        
        # Assign features to patches
        ph4_per_patch = self.assign_features_to_patches(
            mol_ph4_coords,
            mol_ph4_types,
            patch_centers,
            self._patch_size * self.grid_resolution
        )  # [num_patches, num_ph4_types]
        
        # Embed pharmacophore features
        x_ph4 = self._ph4_embedding(ph4_per_patch)  # [batch_size, num_patches, d_model]
        
        # Fusion
        x = self._fusion_ffn(torch.cat([x_shape, x_ph4], dim=-1))
        
        # Add positional embeddings
        if num_patches > self._pos_embed.size(1):
            raise ValueError(f"Sequence length {num_patches} exceeds positional embedding size {self._pos_embed.size(1)}")
        
        x = x + self._pos_embed[:, :num_patches]
        x = self._embed_dropout(x)
        
        # Create padding mask (assuming no padding for now)
        padding_mask = torch.zeros((bz, num_patches), dtype=torch.bool, device=x.device)
        
        # Apply transformer
        x = self._transformer(x)
        x = self._norm(x)
        
        return x, padding_mask

def get_encoder(t: str, cfg) -> BaseEncoder:
    if t == "smiles":
        return SMILESEncoder(**cfg)
    elif t == "graph":
        return GraphEncoder(**cfg)
    elif t == "shape":
        return ShapeEncoderWithPH4(**cfg)  # Updated to use new encoder
    else:
        raise ValueError(f"Unknown encoder type: {t}")
    
    Assign the pharmacofore feature to each shape patch by the center of the patch / hard coordinate boundaries

    def assign_features_to_patches(
    self,
    ph4_coords: torch.Tensor,  # [N_features, 3]
    ph4_types: torch.Tensor,   # [N_features]
    patch_centers: torch.Tensor,  # [num_patches, 3]
    patch_size: float
) -> torch.Tensor:
    """
    Assign pharmacophore features to patches using exact boundary checks
    """
    num_patches = patch_centers.size(0)
    half_size = patch_size / 2
    
    # Initialize feature counts per patch
    patch_features = torch.zeros(
        (num_patches, self.num_ph4_types),
        device=ph4_coords.device
    )
    
    # Calculate patch boundaries
    patch_mins = patch_centers - half_size  # [num_patches, 3]
    patch_maxs = patch_centers + half_size  # [num_patches, 3]
    
    # For each feature
    for feat_idx in range(len(ph4_coords)):
        feat_coord = ph4_coords[feat_idx]  # [3]
        feat_type = ph4_types[feat_idx]    # scalar
        
        # Check if feature is within patch bounds
        in_x = (feat_coord[0] >= patch_mins[:, 0]) & (feat_coord[0] < patch_maxs[:, 0])
        in_y = (feat_coord[1] >= patch_mins[:, 1]) & (feat_coord[1] < patch_maxs[:, 1])
        in_z = (feat_coord[2] >= patch_mins[:, 2]) & (feat_coord[2] < patch_maxs[:, 2])
        
        # Feature must be within bounds in all dimensions
        in_patch = in_x & in_y & in_z  # [num_patches]
        
        # Add feature to relevant patches
        patch_features[in_patch, feat_type] += 1
        
    return patch_features  # [num_patches, num_ph4_types]

Preprocessed:

def _init_shapes_and_ph4(self) -> tuple[dict, dict, dict]:
    shapes_dict = {}
    shape_patches_dict = {}
    ph4_patches_dict = {}
    
    # Get shapes and convert to patches
    shapes = get_shapes_batch(...)
    
    # Get pharmacophore features and assign to patches
    ph4_generator = GaussianPH4Generator()
    
    for mol_idx, (shape, mol) in enumerate(zip(shapes, self._molecules)):
        mol_id = id(mol)
        
        # Shape processing
        shape_patches = get_shape_patches(shape, self._fp_option.patch_size)
        
        # Pharmacophore processing
        ph4_coords, ph4_types = ph4_generator(mol)
        ph4_patches = assign_ph4_to_patches(
            ph4_coords, 
            ph4_types,
            shape.shape,  # grid dimensions
            self._fp_option.patch_size
        )
        
        # Store everything
        shapes_dict[mol_id] = shape
        shape_patches_dict[mol_id] = shape_patches
        ph4_patches_dict[mol_id] = ph4_patches
        
Then added in the encoder as:
    def forward(self, batch: ProjectionBatch) -> tuple[torch.Tensor, torch.Tensor]:
    # Both already in patch form
    shape_patches = batch["shape_patches"]
    ph4_patches = batch["ph4_patches"]
    
    # Process each
    x_shape = self._patch_ffn(shape_patches)
    x_ph4 = self._ph4_embedding(ph4_patches)
    

    Add them at the lowest level of the encoder (same as for the atom/bond features in the graph encoder)
    
    Either with Rotary embeddings
    class ShapeEncoderWithPH4(BaseEncoder):
    def __init__(self, d_model=512, ...):
        # ... other init code ...
        self.rotary = RotaryEmbedding(
            dim=d_model,
            freqs_for="pixel",  # Since we're dealing with 3D spatial data
            max_freq=10
        )
    
    def forward(self, batch):
        x_shape = self._patch_ffn(shape_patches)      # [B, N_patches, d_model]
        x_ph4 = self._ph4_embedding(ph4_patches)      # [B, N_patches, d_model]
        
        # Apply rotary embeddings separately
        x_shape = self.rotary.rotate_queries_or_keys(x_shape)
        x_ph4 = self.rotary.rotate_queries_or_keys(x_ph4)
        
        # Then concatenate
        x = torch.cat([x_shape, x_ph4], dim=1)       # [B, 2*N_patches, d_model]
        
    or single sequence with type tokens
    
    class ShapeEncoderWithPH4(BaseEncoder):
    def __init__(self, d_model=512, ...):
        # ... other init code ...
        self.shape_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.ph4_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoding = PositionalEncoding(d_model)
        
    def forward(self, batch):
        x_shape = self._patch_ffn(shape_patches)      # [B, N_patches, d_model]
        x_ph4 = self._ph4_embedding(ph4_patches)      # [B, N_patches, d_model]
        
        # Add type tokens
        x_shape = x_shape + self.shape_token
        x_ph4 = x_ph4 + self.ph4_token
        
        # Concatenate and add positional encoding
        x = torch.cat([x_shape, x_ph4], dim=1)       # [B, 2*N_patches, d_model]
        x = self.pos_encoding(x)                      # Single positional sequence

     
    or a complete fusion of:
        
        1. Separate Processing with Cross-Attention:
        Process shape and pharmacophore patches separately through their own transformer layers.
        Use cross-attention layers to allow interaction between shape and pharmacophore features.
        2. Rotary Positional Embeddings (RoPE):
        Use RoPE to encode spatial information for both shape and pharmacophore patches.
        This is particularly useful for 3D data, as it maintains spatial relationships.
        3. Type Tokens for Differentiation:
        Add type tokens to distinguish between shape and pharmacophore patches.
        This helps the model understand the context of each patch.
        Late Fusion:
        After separate processing, concatenate the outputs and pass them through a final fusion layer.
        This allows the model to learn combined representations after understanding each modality separately.

    class ShapePh4Encoder(BaseEncoder):
    def __init__(self, d_model=512, nhead=8, num_layers=6, max_seq_length=3000):
        super().__init__()
        self.d_model = d_model
        
        # Separate transformers for shape and pharmacophore
        self.shape_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        self.ph4_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        
        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(d_model, nhead)
        
        # Rotary embeddings
        self.rotary = RotaryEmbedding(dim=d_model, freqs_for="pixel", max_freq=10)
        
        # Type tokens
        self.shape_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.ph4_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Final fusion layer
        self.fusion_ffn = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, shape_patches, ph4_patches):
        # Apply rotary embeddings
        shape_patches = self.rotary.rotate_queries_or_keys(shape_patches)
        ph4_patches = self.rotary.rotate_queries_or_keys(ph4_patches)
        
        # Add type tokens
        shape_patches = shape_patches + self.shape_token
        ph4_patches = ph4_patches + self.ph4_token
        
        # Process separately
        shape_out = self.shape_transformer(shape_patches)
        ph4_out = self.ph4_transformer(ph4_patches)
        
        # Cross-attention
        shape_out, _ = self.cross_attention(shape_out, ph4_out, ph4_out)
        ph4_out, _ = self.cross_attention(ph4_out, shape_out, shape_out)
        
        # Concatenate and fuse
        combined = torch.cat([shape_out, ph4_out], dim=-1)
        fused_output = self.fusion_ffn(combined)
        
        return fused_output
     
     + Morgan fingerprint + E3P fingerprint

Another option:
    Current: both pharm and shape patches in patch form. Cross attention.
    Both, fused, add ACP4 for global feature predisposition pharmacofore relationships and distances:
        Preprocess ACP4 Fingerprints:
        Convert the ACP4 fingerprint into a dense vector representation.
        Use an embedding layer to map the sparse indices to a dense space.
        2. Concatenate with Patch Representations:
        After processing shape and pharmacophore patches, concatenate the ACP4 fingerprint representation.
        3. Process Through a Fusion Layer:
        Use a fusion layer to combine the information from patches and fingerprints.
        
        class ShapePh4ACP4Encoder(BaseEncoder):
    def __init__(self, d_model=512, nhead=8, num_layers=6, max_seq_length=3000, acp4_dim=256):
        super().__init__()
        self.d_model = d_model
        
        # Separate transformers for shape and pharmacophore
        self.shape_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        self.ph4_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        
        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(d_model, nhead)
        
        # Rotary embeddings
        self.rotary = RotaryEmbedding(dim=d_model, freqs_for="pixel", max_freq=10)
        
        # Type tokens
        self.shape_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.ph4_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # ACP4 embedding
        self.acp4_embedding = nn.Linear(acp4_dim, d_model)
        
        # Final fusion layer
        self.fusion_ffn = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, shape_patches, ph4_patches, acp4_fingerprint):
        # Apply rotary embeddings
        shape_patches = self.rotary.rotate_queries_or_keys(shape_patches)
        ph4_patches = self.rotary.rotate_queries_or_keys(ph4_patches)
        
        # Add type tokens
        shape_patches = shape_patches + self.shape_token
        ph4_patches = ph4_patches + self.ph4_token
        
        # Process separately
        shape_out = self.shape_transformer(shape_patches)
        ph4_out = self.ph4_transformer(ph4_patches)
        
        # Cross-attention
        shape_out, _ = self.cross_attention(shape_out, ph4_out, ph4_out)
        ph4_out, _ = self.cross_attention(ph4_out, shape_out, shape_out)
        
        # Embed ACP4 fingerprint
        acp4_out = self.acp4_embedding(acp4_fingerprint)
        
        # Concatenate and fuse
        combined = torch.cat([shape_out, ph4_out, acp4_out.unsqueeze(1)], dim=-1)
        fused_output = self.fusion_ffn(combined)
        
        return fused_output
    

    class FingerprintIndex:
    def __init__(
        self, 
        molecules: Iterable[Molecule], 
        fp_option: FingerprintOption,
        device: torch.device = None
    ) -> None:
        super().__init__()
        self._molecules = tuple(molecules)
        self._fp_option = fp_option
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._fp = self._init_fingerprint(device=self.device)
        self._shapes, self._shape_patches = self._init_shapes()
        # Add ACP4 fingerprints storage
        self._acp4_fingerprints = self._init_acp4_fingerprints()
        self._tree = self._init_tree()
    
    def _init_acp4_fingerprints(self) -> dict[int, np.ndarray]:
        """Initialize ACP4 fingerprints for all molecules"""
        acp4_dict = {}
        for mol_idx, mol in enumerate(self._molecules):
            mol_id = id(mol)
            # Compute ACP4 fingerprint for the molecule
            acp4_fp = mol.get_acp4_fingerprint()  # You'll need to implement this in Molecule class
            acp4_dict[mol_id] = acp4_fp
        return acp4_dict

    @property
    def acp4_fingerprints(self) -> dict[int, np.ndarray]:
        """Access ACP4 fingerprints"""
        return self._acp4_fingerprints

    def get_molecule_features(self, mol_id: int) -> dict:
        """Get all features for a specific molecule"""
        return {
            'shape_patches': self._shape_patches.get(mol_id, []),
            'shapes': self._shapes.get(mol_id, []),
            'acp4_fingerprint': self._acp4_fingerprints.get(mol_id, None)
        }

    def process_molecule_batch(self, mol_batch, atom_stamp):
        """Process a batch of molecules with parallel rotations"""
        results = []
        for mol in mol_batch:
            if mol is None:
                continue
                
            try:
                # ... existing conformer generation code ...
                
                # Get ACP4 fingerprint for this molecule
                mol_id = id(mol)
                acp4_fp = self._acp4_fingerprints.get(mol_id)
                
                # Process all rotations in parallel
                with mp.Pool(processes=min(24, mp.cpu_count())) as rotation_pool:
                    rotation_results = rotation_pool.starmap(
                        self.process_single_rotation,
                        [(mol, rot_mat, atom_stamp, cavity) 
                         for rot_mat in ROTATIONS]
                    )
                
                # Add ACP4 fingerprint to results
                for result in rotation_results:
                    if result is not None:
                        result['acp4_fingerprint'] = acp4_fp
                
                # Collect valid results
                results.extend([r for r in rotation_results if r is not None])
                
            except Exception as e:
                continue
                
        return results

    def process_single_rotation(self, mol, rotation_mat, atom_stamp, cavity, resolution=0.5, box_size=15):
        """Process a single rotation of a molecule with pre-computed conformers"""
        try:
            # ... existing rotation processing code ...
            
            return {
                'mol': mol,
                'shape': centered_shape,
                'shape_patches': shape_patches,
                'mol_id': id(mol)  # Add mol_id to help match with ACP4 fingerprint
            }
            
        except Exception as e:
            print(f"Failed to process rotation: {str(e)}")
            return None
        
    
    projection_dataset.py:
        class ProjectionDataset(Dataset):
    def __getitem__(self, idx):
        mol_idx = self.mol_indices[idx]
        mol_id = id(self._fpindex.molecules[mol_idx])
        
        # Get all features for this molecule
        features = self._fpindex.get_molecule_features(mol_id)
        
        return {
            'shape_patches': features['shape_patches'],
            'pharmacophore_patches': self.get_pharmacophore_patches(mol_idx),
            'acp4_fingerprint': features['acp4_fingerprint'],
            'mol_idx': mol_idx
        }
        
        
BONUS: pharmacofore feature computation!!!! to be exchanges for something more precise perhaps, DO VISUALIZATIONS
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm3D import Pharmacophore

class Molecule:
    def __init__(self, rdmol):
        self._rdmol = rdmol
        # Initialize feature factory
        self.fdef_file = os.path.join(os.path.dirname(__file__), 'BaseFeatures.fdef')
        self.feature_factory = ChemicalFeatures.BuildFeatureFactory(self.fdef_file)
    
    def get_pharmacophore_features(self) -> dict:
        """Get pharmacophore features for the molecule"""
        # Make sure molecule has 3D coordinates
        if not self._rdmol.GetNumConformers():
            raise ValueError("Molecule must have 3D coordinates")
            
        # Get all features
        features = self.feature_factory.GetFeaturesForMol(self._rdmol)
        
        # Organize features by type
        ph4_features = {
            'HBD': [],  # Hydrogen Bond Donor
            'HBA': [],  # Hydrogen Bond Acceptor
            'ARO': [],  # Aromatic
            'HYD': [],  # Hydrophobic
            'POS': [],  # Positive Ionizable
            'NEG': [],  # Negative Ionizable
        }
        
        for feature in features:
            # Get feature position (centroid of atoms in feature)
            atoms = feature.GetAtomIds()
            pos = self._rdmol.GetConformer().GetPositions()[atoms].mean(axis=0)
            
            # Store feature type and position
            feature_type = feature.GetFamily()
            if feature_type in ph4_features:
                ph4_features[feature_type].append({
                    'position': pos,
                    'atoms': atoms,
                    'type': feature_type
                })
        
        return ph4_features

    def get_pharmacophore_grid(self, grid_resolution=0.5, box_size=15) -> np.ndarray:
        """Convert pharmacophore features to a grid representation"""
        features = self.get_pharmacophore_features()
        
        # Create empty grids for each feature type
        grid_size = int(2 * box_size / grid_resolution) + 1
        feature_grids = {
            ftype: np.zeros((grid_size, grid_size, grid_size))
            for ftype in features.keys()
        }
        
        # Center of the grid
        center = np.array([box_size, box_size, box_size])
        
        # Fill grids with gaussian representations of features
        sigma = grid_resolution  # Width of gaussian
        for ftype, feature_list in features.items():
            for feature in feature_list:
                pos = feature['position']
                # Convert position to grid coordinates
                grid_pos = (pos + center) / grid_resolution
                grid_pos = grid_pos.astype(int)
                
                # Add gaussian around feature position
                x = np.arange(grid_size)
                y = np.arange(grid_size)
                z = np.arange(grid_size)
                X, Y, Z = np.meshgrid(x, y, z)
                
                # Gaussian function
                gaussian = np.exp(-((X - grid_pos[0])**2 + 
                                  (Y - grid_pos[1])**2 + 
                                  (Z - grid_pos[2])**2) / (2 * sigma**2))
                
                feature_grids[ftype] += gaussian
        
        # Combine all feature grids into single array
        combined_grid = np.stack([feature_grids[ftype] for ftype in sorted(features.keys())], axis=-1)
        
        return combined_grid

    def get_pharmacophore_patches(self, patch_size=3) -> np.ndarray:
        """Convert pharmacophore grid to patches"""
        ph4_grid = self.get_pharmacophore_grid()
        
        # Convert to patches
        patches = view_as_blocks(ph4_grid, (patch_size, patch_size, patch_size, 6))
        patches = patches.reshape(-1, patch_size**3 * 6)  # 6 feature types
        
        return patches
    
fpindex:

class FingerprintIndex:
    def _init_shapes(self) -> tuple[dict[int, list[np.ndarray]], dict[int, list[np.ndarray]], dict[int, list[np.ndarray]]]:
        shapes_dict = {}
        shape_patches_dict = {}
        ph4_patches_dict = {}
        
        for mol_idx, mol in enumerate(self._molecules):
            mol_id = id(mol)
            
            # Get shape
            shape = get_shape(mol, ...)
            shape_patches = get_shape_patches(shape, self._fp_option.patch_size)
            
            # Get pharmacophore patches
            ph4_patches = mol.get_pharmacophore_patches(patch_size=self._fp_option.patch_size)
            
            shapes_dict[mol_id] = shape
            shape_patches_dict[mol_id] = shape_patches
            ph4_patches_dict[mol_id] = ph4_patches
        
        return shapes_dict, shape_patches_dict, ph4_patches_dict
        
TODO:
[] Run original chemprojector + SBDD
[] Shape encoder Morgan fingeprint generation
[] Run Shape encoder Morgan on training + SBDD
[] Generate Morgan&E3P fingerprints for each molecule and add to fpindex file 
[] Generate ACP4 fingerprints for each molecule and add to fpindex file 
[] Visualize pharmacophore features

[] Shape encoder + pharmacofore features
[] Run Shape encoder + pharmacofore features + Morgan + SBDD

[] Shape encoder + pharmacofore features + ACP4 fingerprint !!!MOST LIKELY THE BEST
[] Run Shape encoder + pharmacofore features + ACP4 fingerprint on training + SBDD
    
[] Shape encoder + pharmacofore features + ACP4 fingerprint + Morgan&E3P fingerprint
    
[] Run Shape encoder + pharmacofore features + ACP4 fingerprint + Morgan&E3P fingerprint on training + SBDD
