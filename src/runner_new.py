#!/usr/bin/env python3
"""
Combined CNN + Markov model for improved character recognition.
Uses Markov top-k as a plausibility filter for CNN predictions.
"""

import os
import sys
import glob
import argparse
import pickle
import subprocess

# Add paths for imports
sys.path.append('src')
sys.path.append('src/markov')


def get_cnn_ranked_predictions(image_path, split="byclass"):
    """Get CNN predictions by calling evaluate.py."""
    result = subprocess.run(
        [sys.executable, "src/evaluate.py", image_path, split, "--nogui", "--top-predictions"],
        capture_output=True,
        text=True,
        cwd=os.getcwd()
    )
    
    if result.returncode != 0:
        print(f"Error running evaluate.py: {result.stderr.strip()}")
        return [("?", 0.0)]
    
    # Parse output (format: char:probability)
    predictions = []
    for line in result.stdout.strip().split('\n'):
        if ':' in line:
            char, prob = line.split(':', 1)
            predictions.append((char.strip(), float(prob.strip())))
    
    if not predictions:
        return [("?", 0.0)]
    
    # Sort by probability descending
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions


def load_markov_model(model_path):
    """Load a trained Markov model."""
    if not os.path.exists(model_path):
        return None
    
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different model formats
    if isinstance(data, dict):
        transitions = data.get('transitions', data.get('model', {}))
        order = data.get('order', 3)
    elif hasattr(data, 'transitions'):
        transitions = data.transitions
        order = getattr(data, 'order', 3)
    elif hasattr(data, 'model'):
        transitions = data.model
        order = getattr(data, 'order', 3)
    else:
        print(f"  Warning: Unknown model format. Keys/attrs: {dir(data) if hasattr(data, '__dict__') else data.keys() if isinstance(data, dict) else type(data)}")
        return None
    
    print(f"  Transitions count: {len(transitions)}")
    
    # Debug: show actual key format
    sample_keys = list(transitions.keys())[:5]
    print(f"  Sample keys: {sample_keys}")
    print(f"  Key type: {type(sample_keys[0]) if sample_keys else 'N/A'}")
    
    # Detect actual order from key length
    if sample_keys and isinstance(sample_keys[0], str):
        detected_order = len(sample_keys[0])
        if detected_order != order:
            print(f"  Note: Detected order {detected_order} from key length (metadata said {order})")
            order = detected_order
    
    return {
        'transitions': transitions,
        'order': order
    }


def get_ensemble_markov_top_k(markov_models, context, k=3, debug=False):
    """Get combined top-k from multiple Markov models."""
    if not markov_models:
        return []
    
    char_scores = {}
    for model in markov_models:
        if model is None:
            continue
        
        order = model['order']
        transitions = model['transitions']
        
        # Try both original case and lowercase
        for ctx_variant in [context, context.lower()]:
            # Get the last N characters as our context
            ctx = ctx_variant[-order:] if len(ctx_variant) >= order else ctx_variant
            
            if debug:
                print(f"    DEBUG: Looking for context ending with '{ctx}' (order={order})")
            
            # If we have enough context, do direct lookup
            if len(ctx) == order and ctx in transitions:
                next_chars = transitions[ctx]
                total = sum(next_chars.values())
                for char, count in next_chars.items():
                    prob = count / total
                    char_scores[char] = char_scores.get(char, 0) + prob
                if debug:
                    print(f"    DEBUG: Direct match '{ctx}' -> {list(next_chars.keys())[:5]}")
                break
            
            # Otherwise, find all keys that end with our context
            if len(ctx) > 0 and len(ctx) < order:
                matching_keys = [key for key in transitions.keys() if key.endswith(ctx)]
                if debug:
                    print(f"    DEBUG: Found {len(matching_keys)} keys ending with '{ctx}'")
                
                if matching_keys:
                    # Aggregate predictions from all matching keys
                    for key in matching_keys:
                        next_chars = transitions[key]
                        total = sum(next_chars.values())
                        for char, count in next_chars.items():
                            prob = count / total
                            # Weight by how much of the context matched
                            char_scores[char] = char_scores.get(char, 0) + prob
                    break
            
            if char_scores:
                break
    
    if not char_scores:
        return []
    
    # Normalize and sort
    total = sum(char_scores.values())
    sorted_chars = sorted(char_scores.items(), key=lambda x: x[1], reverse=True)
    return [char for char, score in sorted_chars[:k]]


def find_best_match(cnn_predictions, markov_top_k, prefer_letters=True, fallback_to_cnn=True):
    """
    Find the highest CNN prediction that's in Markov's top-k.
    Uses case-sensitive matching to avoid I/i matching when l is better.
    If no match and prefer_letters=True, prefer letter predictions over digits.
    """
    if not markov_top_k:
        if prefer_letters:
            # No Markov context yet - prefer letters for text recognition
            for rank, (char, conf) in enumerate(cnn_predictions):
                if char.isalpha():
                    return char, conf, rank, "prefer_letter"
        return cnn_predictions[0][0], cnn_predictions[0][1], 0, "no_markov"
    
    # Case-sensitive matching
    markov_set = set(markov_top_k)
    
    for rank, (char, conf) in enumerate(cnn_predictions):
        if char in markov_set:
            return char, conf, rank, "matched"
    
    # No exact match - try case-insensitive as fallback
    markov_lower = set(c.lower() for c in markov_top_k)
    for rank, (char, conf) in enumerate(cnn_predictions):
        if char.lower() in markov_lower:
            return char, conf, rank, "matched_case_insensitive"
    
    # No match found
    if prefer_letters:
        # Prefer the highest-confidence letter from CNN
        for rank, (char, conf) in enumerate(cnn_predictions):
            if char.isalpha():
                return char, conf, rank, "prefer_letter"
    
    if fallback_to_cnn:
        return cnn_predictions[0][0], cnn_predictions[0][1], 0, "fallback"
    else:
        return markov_top_k[0], 0.0, -1, "markov_only"


def process_images(image_folder, split, markov_models, markov_k=5, prefer_letters=True, verbose=True, debug=False):
    """Process all images with CNN + Markov filter."""
    
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    
    if not image_files:
        print(f"No PNG images found in {image_folder}/")
        return "", []
    
    print(f"Processing {len(image_files)} images...")
    print(f"Markov filter: top-{markov_k} predictions")
    print(f"Prefer letters: {prefer_letters}\n")
    
    predicted_text = ""
    details = []
    first_char_is_letter = None
    
    for i, image_path in enumerate(image_files):
        filename = os.path.basename(image_path)
        
        # Get CNN ranked predictions via evaluate.py
        cnn_predictions = get_cnn_ranked_predictions(image_path, split)
        
        # If first character was a letter, filter to only letters
        if first_char_is_letter is True:
            cnn_predictions_filtered = [(c, p) for c, p in cnn_predictions if c.isalpha()]
            if cnn_predictions_filtered:
                cnn_predictions = cnn_predictions_filtered
        
        cnn_top = cnn_predictions[0]
        
        # Get Markov top-k predictions based on context so far
        markov_top_k = get_ensemble_markov_top_k(markov_models, predicted_text, k=markov_k, debug=debug)
        
        # Find best match
        final_char, final_conf, match_rank, match_type = find_best_match(
            cnn_predictions, markov_top_k, prefer_letters=prefer_letters
        )
        
        # Track if first char is a letter
        if first_char_is_letter is None:
            first_char_is_letter = final_char.isalpha()
            if debug:
                print(f"    DEBUG: First char '{final_char}' is_letter={first_char_is_letter}, filtering future predictions")
        
        predicted_text += final_char
        
        changed = final_char != cnn_top[0]
        detail = {
            'image': filename,
            'cnn_top5': [(c, f"{p:.3f}") for c, p in cnn_predictions[:5]],
            'markov_top_k': markov_top_k,
            'cnn_pred': cnn_top[0],
            'cnn_conf': cnn_top[1],
            'final_pred': final_char,
            'final_conf': final_conf,
            'match_rank': match_rank,
            'match_type': match_type,
            'changed': changed,
            'context': predicted_text[-6:-1] if len(predicted_text) > 1 else "",
            'letters_only': first_char_is_letter
        }
        details.append(detail)
        
        if verbose:
            cnn_str = ', '.join([f"'{c}'" for c, p in cnn_predictions[:5]])
            markov_str = ', '.join([f"'{c}'" for c in markov_top_k]) if markov_top_k else "N/A"
            
            filter_note = " [letters only]" if first_char_is_letter and i > 0 else ""
            
            status = ""
            if changed:
                if match_type == "prefer_letter":
                    status = f" ← PREFER LETTER (CNN #{match_rank+1})"
                elif match_type == "matched_case_insensitive":
                    status = f" ← CHANGED (CNN #{match_rank+1}, case-insensitive)"
                else:
                    status = f" ← CHANGED (CNN #{match_rank+1})"
            elif match_type == "no_markov":
                status = " (no context)"
            elif match_type == "fallback":
                status = " (no match, fallback)"
            elif match_type == "prefer_letter":
                status = " (prefer letter)"
            elif match_type == "matched_case_insensitive":
                status = " (case-insensitive)"
            
            print(f"{filename}:{filter_note} CNN=[{cnn_str}] Markov=[{markov_str}] → '{final_char}'{status}")
    
    return predicted_text, details


def main():
    parser = argparse.ArgumentParser(
        description='Combined CNN + Markov character recognition'
    )
    parser.add_argument('--image-folder', default='testerac_output',
                        help='Folder containing character images')
    parser.add_argument('--split', default='byclass',
                        help='EMNIST split (default: byclass)')
    parser.add_argument('--markov-model', nargs='+', 
                        help='Path(s) to Markov model(s)')
    parser.add_argument('--markov-k', type=int, default=5,
                        help='Number of Markov predictions to consider (default: 5)')
    parser.add_argument('--no-prefer-letters', action='store_true',
                        help='Disable preferring letters over digits when no Markov match')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    parser.add_argument('--debug', action='store_true',
                        help='Show debug info for Markov lookups')
    parser.add_argument('--output', help='Save results to file')
    
    args = parser.parse_args()
    
    # Load Markov model(s)
    markov_models = []
    markov_paths = args.markov_model or [
        'src/markov/final_model.pkl',
    ]
    
    print("Loading Markov models...")
    for path in markov_paths:
        model = load_markov_model(path)
        if model:
            markov_models.append(model)
            print(f"  Loaded: {path} (order {model['order']})")
    
    if not markov_models:
        print("  Warning: No Markov models found. Using CNN only.")
    
    # Process images
    print(f"\n{'='*60}")
    predicted_text, details = process_images(
        args.image_folder, args.split, markov_models, 
        args.markov_k, prefer_letters=not args.no_prefer_letters,
        verbose=not args.quiet, debug=args.debug
    )
    
    # Summary
    print(f"\n{'='*60}")
    print(f"RESULT: {predicted_text}")
    print(f"{'='*60}")
    
    changes = sum(1 for d in details if d['changed'])
    match_ranks = [d['match_rank'] for d in details if d['match_type'] == 'matched']
    
    print(f"\nStatistics:")
    print(f"  Total characters: {len(details)}")
    print(f"  Changed by Markov filter: {changes} ({100*changes/len(details):.1f}%)")
    
    if match_ranks:
        rank_counts = {}
        for r in match_ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        print(f"  Match distribution:")
        for rank in sorted(rank_counts.keys()):
            pct = 100 * rank_counts[rank] / len(details)
            print(f"    CNN #{rank+1}: {rank_counts[rank]} ({pct:.1f}%)")
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"Predicted text: {predicted_text}\n\n")
            f.write(f"Settings: markov_k={args.markov_k}\n\n")
            f.write("Details:\n")
            for d in details:
                ctx = d['context'] if d['context'] else "(start)"
                letters_note = " [letters only]" if d.get('letters_only') else ""
                f.write(f"  {d['image']}:{letters_note} ctx='{ctx}' "
                        f"CNN={d['cnn_top5']} Markov={d['markov_top_k']} "
                        f"→ '{d['final_pred']}' {'(changed)' if d['changed'] else ''}\n")
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()