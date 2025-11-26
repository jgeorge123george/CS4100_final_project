import sys
from torch import save
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from model import ImageClassifier
from data_loader import get_data_loader
from utils import get_device
import matplotlib.pyplot as plt

def train_model(epochs=10, lr=1e-3, device="mps"):
    loss_history = []  # List to track losses
    model = ImageClassifier().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()
    data_loader = get_data_loader()

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
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")

    
    with open('model_state.pt', 'wb') as f: 
        save(model.state_dict(), f)
    print("Training complete! Model saved as 'model_state.pt'.")
    
    # Plot loss curve
    plt.plot(range(1, epochs + 1), loss_history, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig("training_loss_curve.png")
    plt.show()


if __name__ == "__main__":
    # Device setup
    device = get_device()
    epochs_count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f"Using device: {device}")
    train_model(epochs=epochs_count,device=device)