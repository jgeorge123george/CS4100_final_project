import torch
from torch.utils.data import DataLoader
from model import ImageClassifier
from data_loader import get_test_loader, get_num_classes
from utils import get_device
import sys

def evaluate_model(model_path=None, split="byclass", batch_size=32):
    device = get_device()
    print(f"Using device: {device}")
    
    # Default model path based on split
    if model_path is None:
        model_path = f"src/model_{split}.pt"
    
    # Load test data
    test_loader = get_test_loader(batch_size=batch_size, split=split)
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both old and new save formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        num_classes = checkpoint['num_classes']
        saved_split = checkpoint.get('split', split)
        state_dict = checkpoint['model_state_dict']
        print(f"Loaded model trained on '{saved_split}' split with {num_classes} classes")
    else:
        # Old format - assume it's just the state dict
        state_dict = checkpoint
        num_classes = get_num_classes(split)
        print(f"Loaded model (assuming {num_classes} classes for '{split}' split)")
    
    model = ImageClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Accuracy on EMNIST {split} Test Set: {accuracy:.2f}%")
    print(f"Correct: {correct}/{total}")

if __name__ == "__main__":
    split = "balanced"
    model_path = None
    
    if len(sys.argv) > 1:
        split = sys.argv[1]
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    
    evaluate_model(model_path=model_path, split=split)