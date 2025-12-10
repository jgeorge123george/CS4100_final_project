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
    first_char_is_letter = None  # Will be determined from first prediction
    
    for i, image_path in enumerate(image_files):
        print(f"Processing {os.path.basename(image_path)} ({i+1}/{len(image_files)})...", end=" ")
        
        try:
            # For first character, get top predictions to check if we should prioritize letters
            if i == 0:
                result = subprocess.run(
                    [sys.executable, "src/evaluate.py", image_path, split, "--nogui", "--top-predictions"],
                    capture_output=True,
                    text=True,
                    cwd=os.getcwd()
                )
                
                if result.returncode == 0:
                    output_lines = result.stdout.strip().split('\n')
                    # Parse top predictions (format: char:probability)
                    predictions = []
                    for line in output_lines:
                        if ':' in line:
                            char, prob = line.split(':', 1)
                            predictions.append((char, float(prob)))
                    
                    if predictions:
                        # Get the top prediction
                        top_char = predictions[0][0]
                        
                        # If top prediction is a letter, use it and set flag
                        if top_char.isalpha():
                            predicted_char = top_char
                            first_char_is_letter = True
                        else:
                            # Look for highest probability letter
                            best_letter = None
                            best_letter_prob = 0
                            for char, prob in predictions:
                                if char.isalpha() and prob > best_letter_prob:
                                    best_letter = char
                                    best_letter_prob = prob
                            
                            if best_letter:
                                predicted_char = best_letter
                                first_char_is_letter = True
                            else:
                                predicted_char = top_char
                                first_char_is_letter = False
                    else:
                        predicted_char = "?"
                        first_char_is_letter = False
                else:
                    print(f"Error: {result.stderr.strip()}")
                    predicted_char = "?"
                    first_char_is_letter = False
            else:
                # For subsequent characters, use letter priority if first char was a letter
                if first_char_is_letter:
                    result = subprocess.run(
                        [sys.executable, "src/evaluate.py", image_path, split, "--nogui", "--top-predictions"],
                        capture_output=True,
                        text=True,
                        cwd=os.getcwd()
                    )
                    
                    if result.returncode == 0:
                        output_lines = result.stdout.strip().split('\n')
                        # Parse top predictions
                        predictions = []
                        for line in output_lines:
                            if ':' in line:
                                char, prob = line.split(':', 1)
                                predictions.append((char, float(prob)))
                        
                        if predictions:
                            # Look for highest probability letter first
                            best_letter = None
                            best_letter_prob = 0
                            for char, prob in predictions:
                                if char.isalpha() and prob > best_letter_prob:
                                    best_letter = char
                                    best_letter_prob = prob
                            
                            # Use letter if found, otherwise use top prediction
                            predicted_char = best_letter if best_letter else predictions[0][0]
                        else:
                            predicted_char = "?"
                    else:
                        print(f"Error: {result.stderr.strip()}")
                        predicted_char = "?"
                else:
                    # Normal prediction without letter priority
                    result = subprocess.run(
                        [sys.executable, "src/evaluate.py", image_path, split, "--nogui"],
                        capture_output=True,
                        text=True,
                        cwd=os.getcwd()
                    )
                    
                    if result.returncode == 0:
                        predicted_char = result.stdout.strip().split('\n')[-1]
                    else:
                        print(f"Error: {result.stderr.strip()}")
                        predicted_char = "?"
            
            characters.append(predicted_char)
            print(f"→ '{predicted_char}'")
                
        except Exception as e:
            print(f"Exception: {e}")
            characters.append("?")
            if i == 0:
                first_char_is_letter = False
    
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
    parser.add_argument('--split', default='byclass',
                      help='EMNIST split to use (default: byclass)')
    parser.add_argument('--markov-model', 
                      help='Path to Markov model (if not specified, searches for one)')
    parser.add_argument('--length', type=int, default=200,
                      help='Length of text to generate (default: 200)')
    parser.add_argument('--save-chars', default=False, 
                      help='Save extracted characters to file, false by default')
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