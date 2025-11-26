import sys
import os
from PIL import Image, ImageOps
import torch
from torchvision.transforms import ToTensor
from model import ImageClassifier
from data_loader import get_num_classes
import matplotlib.pyplot as plt

def get_device(prefer="mps"):
    """Return a torch.device, preferring MPS, then CUDA, then CPU."""
    # Try MPS first (macOS), if requested and available
    try:
        if prefer == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        # If checking MPS availability throws, ignore and continue
        pass
    # Next try CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Fallback to CPU
    return torch.device("cpu")

def get_emnist_mapping(split="balanced"):
    """Get character mapping for EMNIST splits."""
    # EMNIST balanced uses specific character mappings
    if split == "balanced":
        # 0-9 digits, then A-Z letters (excluding similar looking ones)
        # and a-z for some lowercase letters
        mapping = list(range(10))  # 0-9
        # Add uppercase letters
        mapping.extend([chr(i) for i in range(65, 91)])  # A-Z
        # Add lowercase letters (only those distinct from uppercase in balanced)
        mapping.extend(['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'])
        return mapping[:47]  # Balanced has 47 classes
    elif split == "digits":
        return list(range(10))
    elif split == "letters":
        return [chr(i) for i in range(65, 91)]  # A-Z
    else:
        # For other splits, return index as string
        return list(range(get_num_classes(split)))

def predict_image(image_path, model_path=None, split="balanced", device="mps"):
    # Default model path
    if model_path is None:
        model_path = f"src/model_state_emnist_{split}.pt"
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both old and new save formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        num_classes = checkpoint['num_classes']
        saved_split = checkpoint.get('split', split)
        state_dict = checkpoint['model_state_dict']
        print(f"Using model trained on '{saved_split}' split")
    else:
        state_dict = checkpoint
        num_classes = get_num_classes(split)
        saved_split = split
    
    # Load model
    model = ImageClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Process image
    with ImageOps.invert(Image.open(image_path).convert('L')) as img:
        img = img.resize((28, 28))
        img_tensor = ToTensor()(img).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_index = torch.argmax(output).item()
        confidence = probabilities[0][predicted_index].item() * 100
    
    # Map to character
    mapping = get_emnist_mapping(saved_split)
    if predicted_index < len(mapping):
        predicted_char = str(mapping[predicted_index])
    else:
        predicted_char = str(predicted_index)
    
    # Display result
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title(f"Predicted: {predicted_char} (conf: {confidence:.1f}%)")
    plt.axis('off')
    
    # Show top 5 predictions
    plt.subplot(1, 2, 2)
    top5_probs, top5_indices = torch.topk(probabilities[0], min(5, num_classes))
    top5_chars = []
    for idx in top5_indices:
        if idx < len(mapping):
            top5_chars.append(str(mapping[idx]))
        else:
            top5_chars.append(str(idx.item()))
    
    plt.barh(range(len(top5_chars)), top5_probs.cpu().numpy())
    plt.yticks(range(len(top5_chars)), top5_chars)
    plt.xlabel('Confidence (%)')
    plt.title('Top 5 Predictions')
    plt.tight_layout()
    plt.show()

    print(f"Predicted: {predicted_char} (index: {predicted_index}, confidence: {confidence:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <image_path> [split] [model_path]")
        print("Example: python evaluate.py image.png balanced")
        sys.exit(1)
    
    image_path = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else "balanced"
    model_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        sys.exit(1)

    device = get_device()
    predict_image(image_path, model_path, split, device)