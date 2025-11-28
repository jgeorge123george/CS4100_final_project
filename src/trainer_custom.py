import sys
from torch import save
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from model import ImageClassifier
from custom_data_loader import get_data_loader_custom, get_num_classes_custom
from utils import get_device
import matplotlib.pyplot as plt

def train_model_custom(epochs=10, lr=1e-3, split="letters", device="cpu"):
    """Train model using custom EMNIST data loader."""
    loss_history = []
    
    # Get number of classes for the chosen split
    num_classes = get_num_classes_custom(split)
    print(f"Training on EMNIST '{split}' split with {num_classes} classes")
    
    model = ImageClassifier(num_classes=num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()
    
    # Use custom data loader
    data_loader = get_data_loader_custom(split=split)

    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
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
            batch_count += 1
            
            # Print progress every 100 batches
            if batch_count % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_count}, Loss: {loss.item():.4f}")
        
        epoch_loss = total_loss / len(data_loader)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs} completed, Average Loss: {epoch_loss:.4f}")

    # Save model with split name
    model_filename = f'model_state_emnist_{split}.pt'
    with open(model_filename, 'wb') as f: 
        save(model.state_dict(), f)
    print(f"Training complete! Model saved as '{model_filename}'.")
    
    # Plot loss curve
    plt.plot(range(1, epochs + 1), loss_history, marker='o')
    plt.title(f"EMNIST {split.title()} Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plot_filename = f"training_loss_curve_emnist_{split}.png"
    plt.savefig(plot_filename)
    print(f"Loss curve saved as '{plot_filename}'")
    plt.show()

if __name__ == "__main__":
    # Parse command line arguments
    epochs_count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    split = sys.argv[2] if len(sys.argv) > 2 else "letters"
    
    # Device setup
    device = get_device()
    print(f"Using device: {device}")
    
    train_model_custom(epochs=epochs_count, split=split, device=device)