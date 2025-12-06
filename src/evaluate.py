import sys
import os
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision.transforms import ToTensor
from model import ImageClassifier
from data_loader import get_num_classes
import matplotlib.pyplot as plt

def get_device(prefer="mps"):
    """Return a torch.device, preferring MPS, then CUDA, then CPU."""
    try:
        if prefer == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_emnist_mapping(split="balanced"):
    """Get character mapping for EMNIST splits."""
    if split == "balanced":
        mapping = list(range(10))
        mapping.extend([chr(i) for i in range(65, 91)])
        mapping.extend(['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'])
        return mapping[:47]
    elif split == "byclass":
        # EMNIST byclass has 62 classes (0-9, A-Z, a-z)
        mapping = []
        # 0-9: digits
        mapping.extend([str(i) for i in range(10)])
        # 10-35: uppercase A-Z
        mapping.extend([chr(i) for i in range(65, 91)])
        # 36-61: lowercase a-z
        mapping.extend([chr(i) for i in range(97, 123)])
        return mapping[:62]
    elif split == "digits":
        return list(range(10))
    elif split == "letters":
        return [chr(i) for i in range(65, 91)]
    else:
        return list(range(get_num_classes(split)))

def predict_image(image_path, model_path=None, split="balanced", device="mps", show_gui=True, top_predictions=False):
    """
    Predict the character for image_path using the model.
    If show_gui is False, plotting and extra prints are suppressed and the function
    returns the predicted character string (and prints only that when called from CLI).
    If top_predictions is True, outputs top 5 predictions with probabilities.
    """
    verbose = show_gui

    if model_path is None:
        model_path = f"src/model_{split}.pt"

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        num_classes = checkpoint['num_classes']
        saved_split = checkpoint.get('split', split)
        state_dict = checkpoint['model_state_dict']
        if verbose:
            print(f"Using model trained on '{saved_split}' split")
    else:
        state_dict = checkpoint
        num_classes = get_num_classes(split)
        saved_split = split

    model = ImageClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    with Image.open(image_path).convert('L') as raw_img:
        img = ImageOps.invert(raw_img)
        # Make image square by adding black padding to shorter sides
        width, height = img.size
        if width != height:
            size = max(width, height)
            square_img = Image.new('L', (size, size), 0)
            offset = ((size - width) // 2, (size - height) // 2)
            square_img.paste(img, offset)
            img = square_img
        img = img.resize((28, 28))
        arr = np.array(img)
        if "mnist" in saved_split.lower():
            proc = arr
        else:
            proc = arr.T
        img = Image.fromarray(proc)
        img_tensor = ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_index = torch.argmax(output).item()
        confidence = probabilities[0][predicted_index].item() * 100

    mapping = get_emnist_mapping(saved_split)
    if predicted_index < len(mapping):
        predicted_char = str(mapping[predicted_index])
    else:
        predicted_char = str(predicted_index)

    if show_gui:
        plt.figure(figsize=(8, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"Predicted: {predicted_char} (conf: {confidence:.1f}%)")
        plt.axis('off')

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
    else:
        # When GUI disabled, output based on mode
        if top_predictions:
            # Output top 5 predictions with probabilities
            top5_probs, top5_indices = torch.topk(probabilities[0], min(5, num_classes))
            for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
                if idx < len(mapping):
                    char = str(mapping[idx])
                else:
                    char = str(idx.item())
                print(f"{char}:{prob.item():.4f}")
        else:
            # Only output the predicted character
            print(predicted_char)
        return predicted_char

    return predicted_char

if __name__ == "__main__":
    # Accept flags anywhere in the args
    args = sys.argv[1:]
    if not args:
        print("Usage: python evaluate.py <image_path> [split] [model_path] [--nogui] [--top-predictions]")
        print("Example: python evaluate.py image.png balanced --nogui --top-predictions")
        sys.exit(1)

    nogui = any(a in ("--nogui", "--no-gui") for a in args)
    top_predictions = any(a in ("--top-predictions", "--top") for a in args)
    # remove the flag tokens so positional args remain consistent
    args = [a for a in args if a not in ("--nogui", "--no-gui", "--top-predictions", "--top")]

    image_path = args[0]
    split = args[1] if len(args) > 1 else "balanced"
    model_path = args[2] if len(args) > 2 else None

    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        sys.exit(1)

    device = get_device()
    predict_image(image_path, model_path, split, device, show_gui=not nogui, top_predictions=top_predictions)
