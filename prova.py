import torch
import pickle
from chemprojector.models.encoder import ShapeEncoder
from chemprojector.data.common import ProjectionBatch

def test_shape_encoder():
    # Initialize encoder
    encoder = ShapeEncoder(
        patch_size=3,
        d_model=512,
        nhead=8,
        num_layers=6,
        max_seq_length=343  # (21/3)^3 = 343 patches for 21x21x21 grid
    )
    
    # Load sample shape data
    with open('/itet-stor/sdivita/net_scratch/originale/ChemProjector/data/processed/validation/shape_fingerprint_dataset1.pkl', 'rb') as f:
        shape_data = pickle.load(f)
    
    # Prepare a batch
    shape = shape_data[0]['mol']  # Get first shape
    shape_patches = get_shape_patches(shape, patch_size=3)
    shape_patches = shape_patches.reshape(1, -1, 27)  # Add batch dimension
    shape_patches = torch.tensor(shape_patches, dtype=torch.float32)
    
    # Create batch dictionary
    batch = ProjectionBatch({
        'shape_patches': shape_patches
    })
    
    # Test encoder
    print("Input shape:", shape_patches.shape)
    encoded, mask = encoder(batch)
    print("Encoded shape:", encoded.shape)  # Should be [1, num_patches, 512]
    print("Mask shape:", mask.shape)        # Should be [1, num_patches]
    
    # Test batch processing
    batch_size = 4
    shape_patches_batch = shape_patches.repeat(batch_size, 1, 1)
    batch['shape_patches'] = shape_patches_batch
    
    encoded_batch, mask_batch = encoder(batch)
    print("\nBatch processing:")
    print("Input batch shape:", shape_patches_batch.shape)
    print("Encoded batch shape:", encoded_batch.shape)  # Should be [4, num_patches, 512]
    print("Mask batch shape:", mask_batch.shape)        # Should be [4, num_patches]

if __name__ == "__main__":
    test_shape_encoder()