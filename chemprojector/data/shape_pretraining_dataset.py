class ShapePretrainingDataset:
    def __init__(self, data, grid_resolution=1, max_dist_stamp=3.0, max_dist=10.0, patch_size=3):
        self.data = data
        self.grid_resolution = grid_resolution
        self.max_dist_stamp = max_dist_stamp
        self.max_dist = max_dist
        self.patch_size = patch_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] 