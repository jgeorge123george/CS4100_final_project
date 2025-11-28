#!/usr/bin/env python3
"""
Minimal non-interactive helpers for a trained character-level MarkovModel.
Expect a MarkovModel implementation with:
    - order (int)
    - load_model(path)
    - get_probabilities(state) -> dict char->prob
    - predict_next_char(state, method='weighted', temperature=1.0) -> char or None
"""

import sys
import argparse
from typing import List, Tuple, Dict, Any, Optional
from markov_model import MarkovModel


def load_model(path: str) -> MarkovModel:
        """Load and return a MarkovModel from a file path."""
        m = MarkovModel()
        m.load_model(path)
        return m


def predict_next_chars(model: MarkovModel, input_text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Return the top_k next-character predictions for the given input_text.
        Returns a list of (char, probability) sorted by probability descending.
        If the required state is not found or input_text is too short, returns [].
        """
        if len(input_text) < model.order:
                return []
        state = input_text[-model.order:]
        probs = model.get_probabilities(state)
        if not probs:
                return []
        sorted_items = sorted(probs.items(), key=lambda kv: -kv[1])
        return sorted_items[:top_k]


def autocomplete(model: MarkovModel, input_text: str, n: int = 50, randomness: float = 1.0) -> Dict[str, Any]:
        """
        Autocomplete input_text for up to n characters.
        randomness is passed through as the temperature parameter to predict_next_char.
        Returns a dict:
            {
                'original': input_text,
                'completed': full_text,
                'added': added_suffix,
                'chosen': [char1, char2, ...]  # list of chosen characters
            }
        If input_text is shorter than model.order, returns the original text unchanged.
        """
        if len(input_text) < model.order:
                return {'original': input_text, 'completed': input_text, 'added': '', 'chosen': []}

        generated = input_text
        chosen: List[str] = []
        for _ in range(n):
                state = generated[-model.order:]
                # If randomness == 0, prefer deterministic argmax if model supports it
                method = 'max' if randomness == 0 else 'weighted'
                next_char = model.predict_next_char(state, method=method, temperature=randomness)
                if not next_char:
                        break
                generated += next_char
                chosen.append(next_char)

        return {
                'original': input_text,
                'completed': generated,
                'added': generated[len(input_text):],
                'chosen': chosen
        }


def analyze_model_knowledge(model: MarkovModel, text: str) -> List[Dict[str, Any]]:
        """
        Analyze the model's knowledge for each state in text.
        Returns a flat list of entries of the form:
            {
                'position': i,
                'state': state,
                'char': next_char,
                'prob': probability,
                'rank': rank (1 = highest),
                'is_actual': whether next_char matches text[i+order]
            }
        If a state has no known transitions, a single entry with prob=0 and rank=-1 is emitted:
            {
                'position': i,
                'state': state,
                'char': None,
                'prob': 0.0,
                'rank': -1,
                'is_actual': False
            }
        """
        results: List[Dict[str, Any]] = []
        if len(text) <= model.order:
                return results

        for i in range(len(text) - model.order):
                state = text[i:i + model.order]
                actual_next = text[i + model.order]
                probs = model.get_probabilities(state)
                if not probs:
                        results.append({
                                'position': i,
                                'state': state,
                                'char': None,
                                'prob': 0.0,
                                'rank': -1,
                                'is_actual': False
                        })
                        continue

                sorted_items = sorted(probs.items(), key=lambda kv: -kv[1])
                for rank, (ch, prob) in enumerate(sorted_items, start=1):
                        results.append({
                                'position': i,
                                'state': state,
                                'char': ch,
                                'prob': prob,
                                'rank': rank,
                                'is_actual': (ch == actual_next)
                        })

        return results


def main():
    """Simple command-line interface for testing Markov models."""
    parser = argparse.ArgumentParser(description='Test a trained Markov model')
    parser.add_argument('model', help='Path to trained model file')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict next characters')
    predict_parser.add_argument('text', help='Input text')
    predict_parser.add_argument('-k', '--top-k', type=int, default=5, help='Number of top predictions (default: 5)')
    
    # Complete command
    complete_parser = subparsers.add_parser('complete', help='Autocomplete text')
    complete_parser.add_argument('text', help='Input text to complete')
    complete_parser.add_argument('-n', '--length', type=int, default=50, help='Number of characters to generate (default: 50)')
    complete_parser.add_argument('-r', '--randomness', type=float, default=1.0, help='Randomness/temperature (default: 1.0)')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze model knowledge of text')
    analyze_parser.add_argument('text', help='Text to analyze')
    analyze_parser.add_argument('--show-all', action='store_true', help='Show all predictions (not just actual)')
    
    # Interactive mode
    interactive_parser = subparsers.add_parser('interactive', help='Interactive testing mode')
    
    args = parser.parse_args()
    
    # Load the model
    try:
        model = load_model(args.model)
        print(f"✓ Model loaded: order={model.order}, states={len(model.states):,}\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Execute command
    if args.command == 'predict':
        predictions = predict_next_chars(model, args.text, top_k=args.top_k)
        if not predictions:
            print(f"No predictions available (input too short or unknown state)")
        else:
            state = args.text[-model.order:] if len(args.text) >= model.order else args.text
            print(f"State: '{state}' → next character predictions:\n")
            for i, (char, prob) in enumerate(predictions, 1):
                # Handle special characters for display
                display_char = repr(char)[1:-1] if char in ['\n', '\t', '\r', ' '] else char
                print(f"  {i}. '{display_char}' : {prob:.3f} ({prob*100:.1f}%)")
    
    elif args.command == 'complete':
        result = autocomplete(model, args.text, n=args.length, randomness=args.randomness)
        print(f"Original:  {result['original']}")
        print(f"Completed: {result['completed']}")
        print(f"\nAdded ({len(result['added'])} chars): {result['added']}")
    
    elif args.command == 'analyze':
        analysis = analyze_model_knowledge(model, args.text)
        if not analysis:
            print("Text too short for analysis")
        else:
            # Show summary statistics
            actual_found = [a for a in analysis if a['is_actual'] and a['prob'] > 0]
            print(f"Analysis of '{args.text[:50]}{'...' if len(args.text) > 50 else ''}'")
            print(f"Coverage: {len(actual_found)}/{len(args.text)-model.order} characters predicted\n")
            
            # Show actual predictions
            for entry in analysis:
                if entry['is_actual'] and entry['prob'] > 0:
                    char_display = repr(entry['char'])[1:-1] if entry['char'] in ['\n', '\t', ' '] else entry['char']
                    print(f"✓ [{entry['position']:3d}] '{entry['state']}' → '{char_display}' (p={entry['prob']:.3f}, rank={entry['rank']})")
                elif entry['is_actual'] and entry['prob'] == 0:
                    print(f"✗ [{entry['position']:3d}] '{entry['state']}' → ? (unknown state)")
            
            if args.show_all:
                print("\n--- All predictions ---")
                for entry in analysis[:20]:  # Limit output
                    if entry['char']:
                        char_display = repr(entry['char'])[1:-1] if entry['char'] in ['\n', '\t', ' '] else entry['char']
                        actual = "✓" if entry['is_actual'] else " "
                        print(f"{actual} [{entry['position']:3d}] '{entry['state']}' → '{char_display}' (p={entry['prob']:.3f})")
    
    elif args.command == 'interactive':
        print("=== Interactive Mode ===")
        print("Commands: predict <text>, complete <text>, analyze <text>, quit\n")
        
        while True:
            try:
                user_input = input("> ").strip()
                if user_input.lower() == 'quit':
                    break
                
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("Usage: <command> <text>")
                    continue
                
                cmd, text = parts[0].lower(), parts[1]
                
                if cmd == 'predict':
                    predictions = predict_next_chars(model, text, top_k=5)
                    if predictions:
                        print("Next character predictions:")
                        for char, prob in predictions:
                            char_display = repr(char)[1:-1]
                            print(f"  '{char_display}': {prob:.3f}")
                    else:
                        print("No predictions available")
                
                elif cmd == 'complete':
                    result = autocomplete(model, text, n=30)
                    print(f"→ {result['completed']}")
                
                elif cmd == 'analyze':
                    analysis = analyze_model_knowledge(model, text)
                    actual_found = [a for a in analysis if a['is_actual'] and a['prob'] > 0]
                    print(f"Model knows {len(actual_found)}/{len(text)-model.order} transitions")
                    for entry in analysis[:5]:
                        if entry['is_actual'] and entry['prob'] > 0:
                            print(f"  ✓ '{entry['state']}' → '{entry['char']}' (p={entry['prob']:.3f})")
                
                else:
                    print("Unknown command. Use: predict, complete, analyze, or quit")
                    
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit")
            except Exception as e:
                print(f"Error: {e}")
    
    else:
        # No command specified, show help
        parser.print_help()


if __name__ == "__main__":
    main()