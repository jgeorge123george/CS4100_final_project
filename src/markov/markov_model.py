#!/usr/bin/env python3
"""
Character-level Markov Model for Text Generation
Author: Assistant
Description: Train a Markov model on text data to predict and generate characters
"""

import random
import pickle
import argparse
from collections import defaultdict, Counter
from pathlib import Path


class MarkovModel:
    """Character-level Markov model for text prediction and generation"""
    
    def __init__(self, order=3):
        """
        Initialize Markov model
        
        Args:
            order: number of characters to use as state (context)
        """
        self.order = order
        self.transitions = defaultdict(Counter)
        self.states = set()
        self.char_vocab = set()
    
    def train(self, text, verbose=True):
        """
        Build transition probability table from text
        
        Args:
            text: training text string
            verbose: print training statistics
        """
        # Ensure text is long enough
        if len(text) <= self.order:
            raise ValueError(f"Text too short for order {self.order}")
        
        # Build transition table
        for i in range(len(text) - self.order):
            current_state = text[i:i + self.order]
            next_char = text[i + self.order]
            
            self.transitions[current_state][next_char] += 1
            self.states.add(current_state)
            self.char_vocab.add(next_char)
        
        # Add final state (for completeness)
        self.states.add(text[-self.order:])
        
        # Add all characters to vocabulary
        for char in text:
            self.char_vocab.add(char)
        
        if verbose:
            print(f"Model trained successfully!")
            print(f"  - Order: {self.order}")
            print(f"  - Unique states: {len(self.states):,}")
            print(f"  - Unique transitions: {sum(len(v) for v in self.transitions.values()):,}")
            print(f"  - Character vocabulary size: {len(self.char_vocab)}")
    
    def predict_next_char(self, state, method='weighted', temperature=1.0):
        """
        Predict next character given a state
        
        Args:
            state: current state (string of length 'order')
            method: 'weighted' (probabilistic) or 'max' (most likely)
            temperature: controls randomness (0.5=less random, 2.0=more random)
        
        Returns:
            Predicted character or None if state not found
        """
        if state not in self.transitions:
            return None
        
        char_counts = self.transitions[state]
        
        if method == 'max':
            # Return most frequent character
            return char_counts.most_common(1)[0][0]
        
        elif method == 'weighted':
            # Weighted random selection
            chars = list(char_counts.keys())
            weights = list(char_counts.values())
            
            # Apply temperature
            if temperature != 1.0:
                weights = [w ** (1/temperature) for w in weights]
            
            return random.choices(chars, weights=weights)[0]
    
    def get_probabilities(self, state, smoothing_alpha=0.0):
        """
        Get probability distribution for next character
        
        Args:
            state: current state string
            smoothing_alpha: Laplace smoothing parameter (0 = no smoothing)
        
        Returns:
            Dictionary of character -> probability
        """
        if state not in self.transitions and smoothing_alpha == 0:
            return {}
        
        char_counts = self.transitions[state].copy() if state in self.transitions else Counter()
        
        # Apply smoothing if requested
        if smoothing_alpha > 0:
            for char in self.char_vocab:
                char_counts[char] = char_counts.get(char, 0) + smoothing_alpha
        
        total = sum(char_counts.values())
        if total == 0:
            return {}
        
        return {char: count/total for char, count in char_counts.items()}
    
    def generate_text(self, length=200, seed_text=None, temperature=1.0, method='weighted'):
        """
        Generate text of specified length
        
        Args:
            length: number of characters to generate
            seed_text: optional starting text (must be >= order length)
            temperature: controls randomness (0.5=less random, 2.0=more random)
            method: 'weighted' or 'max'
        
        Returns:
            Generated text string
        """
        # Choose starting state
        if seed_text and len(seed_text) >= self.order:
            current_state = seed_text[-self.order:]
            generated = seed_text
        else:
            current_state = random.choice(list(self.states))
            generated = current_state
        
        for _ in range(length):
            next_char = self.predict_next_char(current_state, method=method, temperature=temperature)
            
            if next_char is None:
                # Dead end - pick random state and continue
                current_state = random.choice(list(self.states))
                generated += " " + current_state  # Add space to separate
            else:
                generated += next_char
                # Update state
                current_state = generated[-self.order:]
        
        return generated
    
    def save_model(self, filepath):
        """Save model to file using pickle"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'order': self.order,
                'transitions': dict(self.transitions),
                'states': self.states,
                'char_vocab': self.char_vocab
            }, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.order = data['order']
            self.transitions = defaultdict(Counter, data['transitions'])
            self.states = data['states']
            self.char_vocab = data.get('char_vocab', set())
        print(f"Model loaded from {filepath}")
    
    def evaluate_perplexity(self, test_text):
        """
        Calculate perplexity on test text (lower is better)
        
        Args:
            test_text: text to evaluate
        
        Returns:
            Perplexity score
        """
        import math
        
        total_log_prob = 0
        count = 0
        
        for i in range(len(test_text) - self.order):
            state = test_text[i:i + self.order]
            next_char = test_text[i + self.order]
            
            probs = self.get_probabilities(state, smoothing_alpha=0.01)
            if next_char in probs and probs[next_char] > 0:
                total_log_prob += math.log(probs[next_char])
                count += 1
        
        if count == 0:
            return float('inf')
        
        avg_log_prob = total_log_prob / count
        perplexity = math.exp(-avg_log_prob)
        return perplexity


def load_text(filepath, encoding='utf-8', normalize_whitespace=False):
    """
    Load and optionally preprocess text from file
    
    Args:
        filepath: path to text file
        encoding: file encoding (default: utf-8)
        normalize_whitespace: whether to normalize whitespace
    
    Returns:
        Processed text string
    """
    with open(filepath, 'r', encoding=encoding) as f:
        text = f.read()
    
    if normalize_whitespace:
        text = ' '.join(text.split())
    
    # Remove any null bytes
    text = text.replace('\0', '')
    
    return text


def interactive_mode(model):
    """Run interactive text generation mode"""
    print("\n=== Interactive Mode ===")
    print("Commands: 'quit' to exit, 'temp <value>' to set temperature")
    print(f"Current settings: order={model.order}, temperature=1.0")
    
    temperature = 1.0
    
    while True:
        user_input = input("\nEnter seed text (or command): ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower().startswith('temp '):
            try:
                temperature = float(user_input.split()[1])
                print(f"Temperature set to {temperature}")
            except:
                print("Invalid temperature value")
            continue
        
        # Generate text
        if len(user_input) < model.order:
            print(f"Seed text too short. Need at least {model.order} characters.")
            user_input = None
        
        try:
            generated = model.generate_text(
                length=200,
                seed_text=user_input if user_input else None,
                temperature=temperature
            )
            print(f"\nGenerated text:\n{generated}")
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Character-level Markov Model for Text Generation')
    parser.add_argument('--train', type=str, help='Path to training text file')
    parser.add_argument('--order', type=int, default=3, help='Model order (default: 3)')
    parser.add_argument('--generate', type=int, help='Generate N characters of text')
    parser.add_argument('--seed', type=str, help='Seed text for generation')
    parser.add_argument('--temperature', type=float, default=1.0, help='Generation temperature (default: 1.0)')
    parser.add_argument('--save', type=str, help='Save trained model to file')
    parser.add_argument('--load', type=str, help='Load model from file')
    parser.add_argument('--interactive', action='store_true', help='Run interactive mode')
    parser.add_argument('--evaluate', type=str, help='Evaluate model on test file')
    
    args = parser.parse_args()
    
    # Create model
    model = MarkovModel(order=args.order)
    
    # Load existing model if specified
    if args.load:
        model.load_model(args.load)
    
    # Train model if training file provided
    if args.train:
        print(f"\nTraining model on: {args.train}")
        text = load_text(args.train)
        print(f"Loaded {len(text):,} characters")
        model.train(text)
        
        # Show sample probabilities
        if len(model.states) > 0:
            sample_state = random.choice(list(model.states))
            probs = model.get_probabilities(sample_state)
            if probs:
                print(f"\nSample probabilities after '{sample_state}':")
                for char, prob in sorted(probs.items(), key=lambda x: -x[1])[:5]:
                    display_char = repr(char) if char in ['\n', '\t', ' '] else char
                    print(f"  {display_char}: {prob:.3f}")
    
    # Save model if requested
    if args.save:
        model.save_model(args.save)
    
    # Evaluate model if test file provided
    if args.evaluate:
        print(f"\nEvaluating on: {args.evaluate}")
        test_text = load_text(args.evaluate)
        perplexity = model.evaluate_perplexity(test_text)
        print(f"Perplexity: {perplexity:.2f}")
    
    # Generate text if requested
    if args.generate:
        if len(model.states) == 0:
            print("Error: No model loaded or trained")
        else:
            print(f"\nGenerating {args.generate} characters...")
            generated = model.generate_text(
                length=args.generate,
                seed_text=args.seed,
                temperature=args.temperature
            )
            print(f"\nGenerated text:\n{generated}")
    
    # Run interactive mode if requested
    if args.interactive:
        if len(model.states) == 0:
            print("Error: No model loaded or trained")
        else:
            interactive_mode(model)


if __name__ == "__main__":
    main()