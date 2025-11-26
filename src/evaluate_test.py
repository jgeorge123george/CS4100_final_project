import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from model import ImageClassifier
from utils import get_device

def evaluate_model(model_path="model_state.pt", batch_size=32):
    device = get_device()
    print(f"Using device: {device}")

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    test_loader = DataLoader(test_data, 32)

    model = ImageClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    print(f"Accuracy on MNIST Test Set: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_model()
