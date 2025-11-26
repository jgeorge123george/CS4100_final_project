import sys
from torch import save
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from model import ImageClassifier
from data_loader import get_data_loader, get_num_classes
from utils import get_device
import matplotlib.pyplot as plt

def train_model(epochs=10, lr=1e-3, split="balanced", device="mps"):
    loss_history = []
    
    # Get number of classes for the chosen split
    num_classes = get_num_classes(split)
    print(f"Training on EMNIST '{split}' split with {num_classes} classes")
    
    model = ImageClassifier(num_classes=num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()
    data_loader = get_data_loader(split=split)

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        epoch_loss = total_loss/len(data_loader)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Save model with split info in filename
    model_filename = f'model_state_emnist_{split}.pt'
    save_dict = {
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'split': split
    }
    with open(model_filename, 'wb') as f: 
        save(save_dict, f)
    print(f"Training complete! Model saved as '{model_filename}'.")
    
    # Plot loss curve
    plt.plot(range(1, epochs + 1), loss_history, marker='o')
    plt.title(f"Training Loss - EMNIST {split}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(f"training_loss_curve_emnist_{split}.png")
    plt.show()


if __name__ == "__main__":
    # Device setup
    device = get_device()
    
    # Parse command line arguments
    epochs_count = 10
    split = "balanced"
    
    if len(sys.argv) > 1:
        epochs_count = int(sys.argv[1])
    if len(sys.argv) > 2:
        split = sys.argv[2]
    
    print(f"Using device: {device}")
    train_model(epochs=epochs_count, split=split, device=device)