class ShapeDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def setup(self, stage=None):
        with open(self.config.shape.dataset_path, 'rb') as f:
            dataset = pickle.load(f)
            
        # Split dataset into train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        ) 