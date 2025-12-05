#!/usr/bin/env python3
"""
Extract characters from images using EMNIST model and generate text using Markov model
"""

import os
import sys
import subprocess
import glob
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

def extract_characters_from_images(image_folder="testerac_output", split="full"):
    """
    Run evaluate.py on all images in the folder and collect predictions.
    
    Args:
        image_folder: Folder containing character images
        split: EMNIST split to use ('full' has most characters)
    
    Returns:
        List of predicted characters in order
    """
    print(f"Extracting characters from images in {image_folder}/...")
    
    # Get all PNG images and sort them by filename
    image_pattern = os.path.join(image_folder, "*.png")
    image_files = sorted(glob.glob(image_pattern))
    
    if not image_files:
        print(f"No PNG images found in {image_folder}/")
        return []
    
    print(f"Found {len(image_files)} images to process")
    
    characters = []
    
    for i, image_path in enumerate(image_files):
        print(f"Processing {os.path.basename(image_path)} ({i+1}/{len(image_files)})...", end=" ")
        
        try:
            # Run evaluate.py with --nogui flag to get just the character
            result = subprocess.run(
                [sys.executable, "src/evaluate.py", image_path, split, "--nogui"],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                # Get the predicted character (last line of output, stripped)
                predicted_char = result.stdout.strip().split('\n')[-1]
                characters.append(predicted_char)
                print(f"→ '{predicted_char}'")
            else:
                print(f"Error: {result.stderr.strip()}")
                characters.append("?")  # Placeholder for failed predictions
                
        except Exception as e:
            print(f"Exception: {e}")
            characters.append("?")
    
    return characters

def load_markov_model(model_path):
    """Load a trained Markov model."""
    try:
        # Import the markov model from the markov subdirectory
        sys.path.append('src/markov')
        from markov_model import MarkovModel
        
        model = MarkovModel()
        model.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading Markov model: {e}")
        return None

def find_markov_model():
    """Find an existing Markov model file."""
    possible_paths = [
        "src/markov/final_model.pkl",
        "src/markov/cc_model.pkl", 
        "src/markov/model.pkl",
        "final_model.pkl",
        "checkpoints/final_model.pkl"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def generate_text_from_characters(characters, model_path=None, length=200):
    """
    Generate text using Markov model seeded with extracted characters.
    
    Args:
        characters: List of characters to use as seed
        model_path: Path to Markov model (if None, tries to find one)
        length: Length of text to generate
    
    Returns:
        Generated text string
    """
    if model_path is None:
        model_path = find_markov_model()
        if model_path is None:
            print("No Markov model found. Please train one first or specify path.")
            return None
    
    model = load_markov_model(model_path)
    if model is None:
        return None
    
    # Join characters into seed text
    seed_text = ''.join(characters)
    print(f"Using seed text: '{seed_text}'")
    
    try:
        generated = model.generate_text(
            length=length,
            seed_text=seed_text,
            temperature=0.8
        )
        return generated
    except Exception as e:
        print(f"Error generating text: {e}")
        return None

def main():
    """Main function to process images and generate text."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract characters and generate text')
    parser.add_argument('--image-folder', default='testerac_output',
                      help='Folder containing character images (default: testerac_output)')
    parser.add_argument('--split', default='full',
                      help='EMNIST split to use (default: full)')
    parser.add_argument('--markov-model', 
                      help='Path to Markov model (if not specified, searches for one)')
    parser.add_argument('--length', type=int, default=200,
                      help='Length of text to generate (default: 200)')
    parser.add_argument('--save-chars', 
                      help='Save extracted characters to file')
    parser.add_argument('--load-chars',
                      help='Load characters from file instead of processing images')
    
    args = parser.parse_args()
    
    # Step 1: Extract characters from images (or load from file)
    if args.load_chars and os.path.exists(args.load_chars):
        print(f"Loading characters from {args.load_chars}")
        with open(args.load_chars, 'r', encoding='utf-8') as f:
            characters = list(f.read().strip())
    else:
        characters = extract_characters_from_images(args.image_folder, args.split)
    
    if not characters:
        print("No characters extracted!")
        return
    
    # Save characters if requested
    if args.save_chars:
        with open(args.save_chars, 'w', encoding='utf-8') as f:
            f.write(''.join(characters))
        print(f"Saved {len(characters)} characters to {args.save_chars}")
    
    # Display results
    print(f"\n=== Extracted Characters ===")
    print(f"Total characters: {len(characters)}")
    print(f"Characters: {''.join(characters)}")
    
    # Step 2: Generate text using Markov model
    print(f"\n=== Generating Text ===")
    generated_text = generate_text_from_characters(
        characters, 
        args.markov_model, 
        args.length
    )
    
    if generated_text:
        print(f"\n=== Generated Text ({len(generated_text)} chars) ===")
        print(generated_text)
        
        # Save generated text
        output_file = "generated_text.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Seed: {''.join(characters)}\n")
            f.write(f"Generated:\n{generated_text}")
        print(f"\nSaved generated text to {output_file}")
    else:
        print("Failed to generate text.")

if __name__ == "__main__":
    main()