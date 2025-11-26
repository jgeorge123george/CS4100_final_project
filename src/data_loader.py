from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def get_data_loader(batch_size=32, split="balanced"):
    """
    Get EMNIST data loader.
    
    Args:
        batch_size: Batch size for DataLoader
        split: EMNIST split to use. Options:
            - "balanced": 47 balanced classes (digits + letters)
            - "byclass": 62 unbalanced classes (digits + uppercase + lowercase)
            - "bymerge": 47 unbalanced classes
            - "digits": 10 digit classes only
            - "letters": 26 letter classes only
            - "mnist": Original MNIST subset
    """
    train_data = datasets.EMNIST(
        root="data", 
        split=split,
        train=True, 
        download=True, 
        transform=ToTensor()
    )
    return DataLoader(train_data, batch_size=batch_size, shuffle=True)

def get_test_loader(batch_size=32, split="balanced"):
    """Get EMNIST test data loader."""
    test_data = datasets.EMNIST(
        root="data",
        split=split, 
        train=False,
        download=True,
        transform=ToTensor()
    )
    return DataLoader(test_data, batch_size=batch_size)

def get_num_classes(split="balanced"):
    """Return number of classes for each EMNIST split."""
    class_counts = {
        "balanced": 47,
        "byclass": 62,
        "bymerge": 47,
        "digits": 10,
        "letters": 26,
        "mnist": 10
    }
    return class_counts.get(split, 47)