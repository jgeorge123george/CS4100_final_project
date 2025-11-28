from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
import struct
import numpy as np
import torch
import os

class CustomEMNISTDataset(Dataset):
    """Custom EMNIST dataset that reads directly from raw files."""
    
    def __init__(self, root, split='letters', train=True):
        self.root = root
        self.split = split
        self.train = train
        self.data, self.targets = self._load_data()
        
    def _load_data(self):
        """Load data from the raw EMNIST files."""
        if self.train:
            images_file = f'emnist-{self.split}-train-images-idx3-ubyte'
            labels_file = f'emnist-{self.split}-train-labels-idx1-ubyte'
        else:
            images_file = f'emnist-{self.split}-test-images-idx3-ubyte'
            labels_file = f'emnist-{self.split}-test-labels-idx1-ubyte'
            
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
        image = image.unsqueeze(0) / 255.0  # Add channel dimension and normalize
        return image, target

def get_data_loader(batch_size=32, split="balanced"):
    """
    Get EMNIST data loader with fallback to custom loader, or MNIST if specified.
    """
    # Handle MNIST dataset separately
    if split == "pure_mnist":
        try:
            train_data = datasets.MNIST(
                root="data",
                train=True,
                download=True,
                transform=ToTensor()
            )
            print(f"Using PyTorch MNIST dataset: {len(train_data)} training samples")
            return DataLoader(train_data, batch_size=batch_size, shuffle=True)
        except Exception as e:
            print(f"Failed to load MNIST: {e}")
            raise
    
    # Try PyTorch's EMNIST loader first
    try:
        train_data = datasets.EMNIST(
            root="data", 
            split=split,
            train=True, 
            download=False, 
            transform=ToTensor()
        )
        print(f"Using PyTorch EMNIST loader for '{split}' split")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"PyTorch EMNIST loader failed, using custom loader for '{split}' split...")
        try:
            train_data = CustomEMNISTDataset(
                root="data",
                split=split,
                train=True
            )
            print(f"Custom loader successful: {len(train_data)} training samples")
        except Exception as custom_e:
            print(f"Custom loader also failed: {custom_e}")
            raise RuntimeError(f"Both PyTorch and custom loaders failed for split '{split}'")
    
    return DataLoader(train_data, batch_size=batch_size, shuffle=True)

def get_test_loader(batch_size=32, split="balanced"):
    """Get EMNIST test data loader with fallback to custom loader, or MNIST if specified."""
    # Handle MNIST dataset separately  
    if split == "pure_mnist":
        try:
            test_data = datasets.MNIST(
                root="data",
                train=False,
                download=True,
                transform=ToTensor()
            )
            print(f"Using PyTorch MNIST test dataset: {len(test_data)} test samples")
            return DataLoader(test_data, batch_size=batch_size)
        except Exception as e:
            print(f"Failed to load MNIST test set: {e}")
            raise
    
    try:
        test_data = datasets.EMNIST(
            root="data",
            split=split, 
            train=False,
            download=False,
            transform=ToTensor()
        )
        print(f"Using PyTorch EMNIST test loader for '{split}' split")
    except (FileNotFoundError, RuntimeError):
        print(f"PyTorch EMNIST test loader failed, using custom loader for '{split}' split...")
        try:
            test_data = CustomEMNISTDataset(
                root="data",
                split=split,
                train=False
            )
            print(f"Custom test loader successful: {len(test_data)} test samples")
        except Exception as custom_e:
            print(f"Custom test loader also failed: {custom_e}")
            raise RuntimeError(f"Both PyTorch and custom test loaders failed for split '{split}'")
    
    return DataLoader(test_data, batch_size=batch_size)

def get_num_classes(split="balanced"):
    """Return number of classes for each EMNIST split."""
    class_counts = {
        "balanced": 47,
        "byclass": 62,
        "bymerge": 47,
        "digits": 10,
        "letters": 37,  # EMNIST letters actually has 37 classes
        "mnist": 10,
        "pure_mnist": 10  # Original MNIST dataset
    }
    return class_counts.get(split, 47)