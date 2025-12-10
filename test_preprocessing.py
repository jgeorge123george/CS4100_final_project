import sys
import os
from PIL import Image, ImageOps
import torch
from torchvision.transforms import ToTensor
import numpy as np

sys.path.append('src')
from model import ImageClassifier
from data_loader import get_num_classes
from evaluate import get_emnist_mapping, get_device

def predict_image_corrected(image_path, model_path=None, split="balanced", device="cpu"):
    """Predict image with correct EMNIST preprocessing."""
    
    # Default model path
    if model_path is None:
        model_path = f"src/model_{split}.pt"
    
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

    # Process image with EMNIST-style preprocessing
    original_img = Image.open(image_path).convert('L')
    
    # Try different preprocessing approaches
    preprocessing_methods = {
        "current": lambda img: ImageOps.invert(img).resize((28, 28)),
        "transpose": lambda img: Image.fromarray(np.array(ImageOps.invert(img).resize((28, 28))).T),
        "rotate90": lambda img: ImageOps.invert(img).resize((28, 28)).rotate(90),
        "rotate180": lambda img: ImageOps.invert(img).resize((28, 28)).rotate(180), 
        "rotate270": lambda img: ImageOps.invert(img).resize((28, 28)).rotate(270),
        "transpose_rotate90": lambda img: Image.fromarray(np.rot90(np.array(ImageOps.invert(img).resize((28, 28))).T, 1)),
    }
    
    results = {}
    
    for method_name, preprocess_func in preprocessing_methods.items():
        processed_img = preprocess_func(original_img)
        img_tensor = ToTensor()(processed_img).unsqueeze(0).to(device)
        
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
        
        results[method_name] = {
            'char': predicted_char,
            'confidence': confidence,
            'index': predicted_index
        }
        
        print(f"{method_name:20}: {predicted_char} (confidence: {confidence:.1f}%)")
    
    return results

if __name__ == "__main__":
    device = get_device("cpu")
    
    # Test all your images
    test_images = ['test_char_0.png', 'test_char_1.png', 'test_char_2.png', 'test_char_3.png']
    expected_chars = ['3', 'c', 'd', 'x']  # Based on your description
    
    for i, img_path in enumerate(test_images):
        if os.path.exists(img_path):
            print(f"\n=== Testing {img_path} (expected: '{expected_chars[i]}') ===")
            results = predict_image_corrected(img_path, split="balanced", device=device)
        else:
            print(f"File {img_path} not found")