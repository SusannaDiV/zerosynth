import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_center_region(shape, target_size=21):
    """
    Extract the central region of the shape
    shape: numpy array of shape [N, N, N]
    target_size: size of the region to extract (default: 21)
    """
    if shape.shape[0] == target_size:
        return shape
        
    start_idx = (shape.shape[0] - target_size) // 2
    end_idx = start_idx + target_size
    return shape[start_idx:end_idx, start_idx:end_idx, start_idx:end_idx]

def visualize_shape(shape, save_path=None):
    """
    Visualize a molecular shape using matplotlib
    shape: numpy array of shape [21, 21, 21]
    save_path: path to save the visualization
    """
    # Extract central region
    shape = extract_center_region(shape)
    
    # Create figure with subplots for different views
    fig = plt.figure(figsize=(20, 5))
    
    # XY view (top)
    ax1 = fig.add_subplot(141)
    ax1.imshow(np.max(shape, axis=2), cmap='viridis')
    ax1.set_title('Top View (XY)')
    
    # XZ view (front)
    ax2 = fig.add_subplot(142)
    ax2.imshow(np.max(shape, axis=1), cmap='viridis')
    ax2.set_title('Front View (XZ)')
    
    # YZ view (side)
    ax3 = fig.add_subplot(143)
    ax3.imshow(np.max(shape, axis=0), cmap='viridis')
    ax3.set_title('Side View (YZ)')
    
    # 3D view
    ax4 = fig.add_subplot(144, projection='3d')
    x, y, z = np.where(shape > 0.5)  # Show points where density is > 0.5
    scatter = ax4.scatter(x, y, z, c=shape[x, y, z], cmap='viridis')
    ax4.set_title('3D View')
    plt.colorbar(scatter, ax=ax4)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    # Create visualization directory
    viz_dir = 'visualization'
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load the shape dataset
    data_path = '/itet-stor/sdivita/net_scratch/originale/ChemProjector/chemprojector/data/processed/all/shape_dataset.pkl'
    print(f"Loading shape dataset from {data_path}")
    
    with open(data_path, 'rb') as f:
        dataset = pkl.load(f)
    
    print(f"Dataset contains {len(dataset.data)} shapes")
    
    # Visualize first few shapes
    for i, sample in enumerate(dataset.data[:5]):  # First 5 shapes
        shape = sample['mol']  # Assuming this is the shape array
        print(f"Processing shape {i} with dimensions {shape.shape}")
        
        save_path = os.path.join(viz_dir, f'molecule_shape_{i}.png')
        print(f"Saving visualization {i} to {save_path}")
        
        try:
            visualize_shape(shape, save_path=save_path)
            print(f"Successfully saved visualization for shape {i}")
        except Exception as e:
            print(f"Failed to visualize shape {i}: {e}")

if __name__ == "__main__":
    main() 