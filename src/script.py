import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor

class ImageClassifier(nn.Module):
    """Exact CNN model structure from your model.py"""
    def __init__(self, num_classes=47):
        super().__init__()
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3), 
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 22 * 22, num_classes)       
        )

    def forward(self, x):
        return self.model(x)

def get_feature_progression_array(image_path, model_path="src/model_state_emnist_balanced.pt"):
    """
    Get an array of feature map images showing CNN progression.
    
    Args:
        image_path: Path to input image
        model_path: Path to trained model
    
    Returns:
        images: List of numpy arrays, each representing a stage of processing
                [original, conv1+relu, conv2+relu, conv3+relu]
    """
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint.get('num_classes', 47) if isinstance(checkpoint, dict) else 47
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) else checkpoint
    
    model = ImageClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load and preprocess image
    with Image.open(image_path).convert('L') as img:
        img = ImageOps.invert(img)
        img = img.resize((28, 28))
        img_array = np.array(img)
        
        # Transpose for EMNIST (not MNIST)
        if "mnist" not in model_path.lower():
            img_array = img_array.T
            
        img_tensor = ToTensor()(Image.fromarray(img_array)).unsqueeze(0).to(device)
    
    # Collect feature maps
    images = []
    
    # Add original image
    images.append(img_tensor[0, 0].cpu().numpy())
    
    # Forward pass through layers, collecting outputs
    x = img_tensor
    with torch.no_grad():
        # Process through each layer in the Sequential model
        for i, layer in enumerate(model.model):
            x = layer(x)
            
            # After each ReLU (indices 1, 3, 5), save the output
            if i in [1, 3, 5]:  # After ReLU layers
                # Take the mean across all channels for visualization
                if len(x.shape) == 4:  # Conv output (batch, channels, height, width)
                    # Option 1: Show first channel
                    feature_map = x[0, 0].cpu().numpy()
                    
                    # Option 2: Show average of all channels (uncomment if preferred)
                    # feature_map = x[0].mean(dim=0).cpu().numpy()
                    
                    # Option 3: Show channel with highest activation (uncomment if preferred)
                    # channel_activations = x[0].abs().mean(dim=(1,2))
                    # best_channel = channel_activations.argmax()
                    # feature_map = x[0, best_channel].cpu().numpy()
                    
                    images.append(feature_map)
            
            # Stop after flattening (we don't need FC layer output)
            if isinstance(layer, nn.Flatten):
                break
    
    return images

def display_feature_array(images):
    """Display the array of feature maps."""
    import matplotlib.pyplot as plt
    
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(3*num_images, 3))
    
    if num_images == 1:
        axes = [axes]
    
    titles = ["Original (28×28)", "Conv1+ReLU (26×26×32)", "Conv2+ReLU (24×24×64)", "Conv3+ReLU (22×22×64)"]
    
    for i, (img, ax) in enumerate(zip(images, axes)):
        ax.imshow(img, cmap='gray')
        ax.set_title(titles[i] if i < len(titles) else f"Layer {i}", fontsize=10)
        ax.axis('off')
    
    plt.suptitle("CNN Feature Progression", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return fig

def save_feature_maps_as_images(images, output_dir="feature_maps"):
    """Save each feature map as a separate image file."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    names = ["00_original", "01_conv1_relu", "02_conv2_relu", "03_conv3_relu"]
    
    for i, (img, name) in enumerate(zip(images, names)):
        # Normalize to 0-255
        img_normalized = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
        
        # Save as PNG
        pil_img = Image.fromarray(img_normalized, mode='L')
        pil_img.save(os.path.join(output_dir, f"{name}.png"))
    
    print(f"Saved {len(images)} feature maps to {output_dir}/")

# Quick usage example
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path> [model_path]")
        print("Example: python script.py testerac_output/char_0.png")
    else:
        image_path = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 else "src/model_state_emnist_balanced.pt"
        
        # Get the array of images
        feature_array = get_feature_progression_array(image_path, model_path)
        
        print(f"Generated {len(feature_array)} feature maps")
        print(f"Shapes: {[img.shape for img in feature_array]}")
        
        # Save as individual images
        save_feature_maps_as_images(feature_array)
        
        # Display them
        display_feature_array(feature_array)