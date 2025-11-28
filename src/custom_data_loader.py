import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import gzip
import os

class CustomEMNISTDataset(Dataset):
    """Custom EMNIST dataset that reads directly from the raw files."""
    
    def __init__(self, root, split='letters', train=True, transform=None):
        self.root = root
        self.split = split
        self.train = train
        self.transform = transform
        
        # Load the data
        self.data, self.targets = self._load_data()
        
    def _load_data(self):
        """Load data from the raw EMNIST files."""
        if self.train:
            images_file = f'emnist-{self.split}-train-images-idx3-ubyte'
            labels_file = f'emnist-{self.split}-train-labels-idx1-ubyte'
        else:
            images_file = f'emnist-{self.split}-test-images-idx3-ubyte'
            labels_file = f'emnist-{self.split}-test-labels-idx1-ubyte'
            
        # Data is in src/data directory 
        current_dir = os.path.dirname(os.path.abspath(__file__))
        images_path = os.path.join(current_dir, self.root, 'EMNIST', 'raw', images_file)
        labels_path = os.path.join(current_dir, self.root, 'EMNIST', 'raw', labels_file)
        
        # Load images
        with open(images_path, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols).copy()
            
        # Load labels
        with open(labels_path, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8).copy()
            
        return torch.from_numpy(images).float(), torch.from_numpy(labels).long()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, target = self.data[idx], self.targets[idx]
        
        # Convert to PIL Image format if transform expects it
        image = image.unsqueeze(0)  # Add channel dimension
        image = image / 255.0  # Normalize to [0, 1]
        
        if self.transform:
            image = self.transform(image)
            
        return image, target

def get_data_loader_custom(batch_size=32, split="letters"):
    """
    Get custom EMNIST data loader that reads directly from raw files.
    
    Args:
        batch_size: Batch size for DataLoader
        split: EMNIST split to use ('letters', 'balanced', 'digits', etc.)
    """
    try:
        train_data = CustomEMNISTDataset(
            root="data",  # Data is in src/data
            split=split,
            train=True, 
            transform=None
        )
        print(f"Loaded {len(train_data)} training samples for '{split}' split")
        return DataLoader(train_data, batch_size=batch_size, shuffle=True)
    except Exception as e:
        print(f"Error loading custom dataset: {e}")
        raise

def get_test_loader_custom(batch_size=32, split="letters"):
    """Get custom EMNIST test data loader."""
    try:
        test_data = CustomEMNISTDataset(
            root="data",  # Data is in src/data
            split=split, 
            train=False,
            transform=None
        )
        print(f"Loaded {len(test_data)} test samples for '{split}' split")
        return DataLoader(test_data, batch_size=batch_size)
    except Exception as e:
        print(f"Error loading custom test dataset: {e}")
        raise

def get_num_classes_custom(split="letters"):
    """Return number of classes for each EMNIST split."""
    class_counts = {
        "balanced": 47,
        "byclass": 62,
        "bymerge": 47,
        "digits": 10,
        "letters": 37,  # EMNIST letters actually has 37 classes (N_CLASSES field)
        "mnist": 10
    }
    return class_counts.get(split, 37)