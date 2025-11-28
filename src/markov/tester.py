#!/usr/bin/env python3
"""
Interactive Tester for Character-level Markov Model
Author: Assistant
Description: Test a trained Markov model by providing input and getting predictions
"""

import sys
import argparse
from pathlib import Path

# Import the MarkovModel
try:
    from markov_model import MarkovModel
except ImportError:
    print("Error: markov_model.py not found in current directory")
    sys.exit(1)


class MarkovTester:
    """Interactive tester for Markov models"""
    
    def __init__(self, model_path):
        """Load a trained model"""
        self.model = MarkovModel()
        self.model.load_model(model_path)
        print(f"\nModel loaded successfully!")
        print(f"  - Order: {self.model.order}")
        print(f"  - States: {len(self.model.states):,}")
        print(f"  - Vocabulary size: {len(self.model.char_vocab)}")
    
    def predict_single_char(self, input_text):
        """Predict the next single character"""
        if len(input_text) < self.model.order:
            return f"Error: Need at least {self.model.order} characters (got {len(input_text)})"
        
        state = input_text[-self.model.order:]
        
        # Get probabilities
        probs = self.model.get_probabilities(state)
        
        if not probs:
            return f"No predictions available for '{state}'"
        
        # Get top predictions
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        
        # Get single best prediction
        best_char = self.model.predict_next_char(state, method='weighted')
        
        return {
            'state': state,
            'best_prediction': best_char,
            'top_5_predictions': sorted_probs[:5],
            'total_options': len(probs)
        }
    
    def predict_multiple_chars(self, input_text, num_chars=10, temperature=1.0, method='weighted'):
        """Predict multiple characters continuing from input"""
        if len(input_text) < self.model.order:
            return f"Error: Need at least {self.model.order} characters (got {len(input_text)})"
        
        generated = input_text
        predictions = []
        
        for _ in range(num_chars):
            state = generated[-self.model.order:]
            next_char = self.model.predict_next_char(state, method=method, temperature=temperature)
            
            if next_char is None:
                predictions.append({'char': '<?>', 'state': state, 'found': False})
                # Try to recover by using a random state
                import random
                generated = generated + random.choice(list(self.model.states))
            else:
                predictions.append({'char': next_char, 'state': state, 'found': True})
                generated += next_char
        
        return {
            'original_input': input_text,
            'generated_text': generated,
            'predictions': predictions,
            'continuation_only': generated[len(input_text):]
        }
    
    def autocomplete(self, input_text, max_length=50, temperature=0.8, stop_chars=None):
        """Autocomplete text until reaching a stop character or max length"""
        if stop_chars is None:
            stop_chars = ['.', '!', '?', '\n']
        
        if len(input_text) < self.model.order:
            return f"Error: Need at least {self.model.order} characters (got {len(input_text)})"
        
        generated = input_text
        
        for _ in range(max_length):
            state = generated[-self.model.order:]
            next_char = self.model.predict_next_char(state, method='weighted', temperature=temperature)
            
            if next_char is None:
                break
            
            generated += next_char
            
            if next_char in stop_chars:
                break
        
        return {
            'input': input_text,
            'completed': generated,
            'added': generated[len(input_text):]
        }
    
    def analyze_context(self, text):
        """Analyze what the model knows about a piece of text"""
        if len(text) < self.model.order:
            return f"Error: Need at least {self.model.order} characters (got {len(text)})"
        
        analysis = []
        
        for i in range(len(text) - self.model.order):
            state = text[i:i + self.model.order]
            actual_next = text[i + self.model.order]
            
            probs = self.model.get_probabilities(state)
            
            if actual_next in probs:
                prob = probs[actual_next]
                rank = sorted(probs.values(), reverse=True).index(prob) + 1
                analysis.append({
                    'position': i,
                    'state': state,
                    'actual': actual_next,
                    'probability': prob,
                    'rank': rank,
                    'total_options': len(probs)
                })
            else:
                analysis.append({
                    'position': i,
                    'state': state,
                    'actual': actual_next,
                    'probability': 0,
                    'rank': -1,
                    'total_options': 0
                })
        
        # Calculate average probability
        valid_probs = [a['probability'] for a in analysis if a['probability'] > 0]
        avg_prob = sum(valid_probs) / len(valid_probs) if valid_probs else 0
        
        return {
            'text': text,
            'analysis': analysis,
            'average_probability': avg_prob,
            'coverage': len(valid_probs) / len(analysis) if analysis else 0
        }


def interactive_mode(tester):
    """Run fully interactive testing mode"""
    print("\n" + "="*60)
    print("INTERACTIVE MARKOV MODEL TESTER")
    print("="*60)
    
    print("\nCommands:")
    print("  predict <text>     - Predict next character")
    print("  continue <text>    - Continue text for 10 characters")
    print("  complete <text>    - Autocomplete until punctuation")
    print("  analyze <text>     - Analyze model's knowledge of text")
    print("  temp <value>       - Set temperature (current: 1.0)")
    print("  help              - Show this help")
    print("  quit              - Exit")
    
    temperature = 1.0
    
    while True:
        print("\n" + "-"*40)
        user_input = input("Enter command: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        elif user_input.lower() == 'help':
            print("\nCommands:")
            print("  predict <text>     - Predict next character with probabilities")
            print("  continue <text>    - Continue text for 10 characters")
            print("  complete <text>    - Autocomplete until punctuation")
            print("  analyze <text>     - Analyze model's knowledge of text")
            print("  temp <value>       - Set temperature (0.5=focused, 2.0=creative)")
            print("  quit              - Exit")
        
        elif user_input.lower().startswith('temp '):
            try:
                temperature = float(user_input.split(maxsplit=1)[1])
                print(f"Temperature set to {temperature}")
            except:
                print("Invalid temperature value")
        
        elif user_input.lower().startswith('predict '):
            text = user_input[8:]  # Remove 'predict '
            if text:
                result = tester.predict_single_char(text)
                if isinstance(result, str):
                    print(result)
                else:
                    print(f"\nInput: '{text}'")
                    print(f"State: '{result['state']}' → '{result['best_prediction']}'")
                    print(f"\nTop 5 predictions:")
                    for char, prob in result['top_5_predictions']:
                        display_char = repr(char)[1:-1]  # Remove quotes from repr
                        print(f"  '{display_char}': {prob:.3f} ({prob*100:.1f}%)")
                    print(f"Total options: {result['total_options']}")
        
        elif user_input.lower().startswith('continue '):
            text = user_input[9:]  # Remove 'continue '
            if text:
                result = tester.predict_multiple_chars(text, num_chars=10, temperature=temperature)
                if isinstance(result, str):
                    print(result)
                else:
                    print(f"\nInput: '{text}'")
                    print(f"Generated: '{result['generated_text']}'")
                    print(f"Added: '{result['continuation_only']}'")
                    
                    # Show character-by-character predictions
                    print(f"\nCharacter-by-character:")
                    for pred in result['predictions']:
                        status = "✓" if pred['found'] else "✗"
                        display_char = repr(pred['char'])[1:-1]
                        print(f"  [{status}] '{pred['state']}' → '{display_char}'")
        
        elif user_input.lower().startswith('complete '):
            text = user_input[9:]  # Remove 'complete '
            if text:
                result = tester.autocomplete(text, temperature=temperature)
                if isinstance(result, str):
                    print(result)
                else:
                    print(f"\nInput: '{text}'")
                    print(f"Completed: '{result['completed']}'")
                    print(f"Added: '{result['added']}'")
        
        elif user_input.lower().startswith('analyze '):
            text = user_input[8:]  # Remove 'analyze '
            if text:
                result = tester.analyze_context(text)
                if isinstance(result, str):
                    print(result)
                else:
                    print(f"\nAnalyzing: '{text}'")
                    print(f"Average probability: {result['average_probability']:.3f}")
                    print(f"Coverage: {result['coverage']*100:.1f}%")
                    
                    print(f"\nDetailed analysis:")
                    for a in result['analysis'][:10]:  # Show first 10
                        display_actual = repr(a['actual'])[1:-1]
                        if a['probability'] > 0:
                            print(f"  '{a['state']}' → '{display_actual}': "
                                  f"p={a['probability']:.3f}, rank={a['rank']}/{a['total_options']}")
                        else:
                            print(f"  '{a['state']}' → '{display_actual}': NOT FOUND")
        
        else:
            print("Unknown command. Type 'help' for commands.")


def main():
    parser = argparse.ArgumentParser(description='Test a trained Markov model')
    parser.add_argument('model', type=str, help='Path to trained model file')
    parser.add_argument('--text', type=str, help='Text to predict from')
    parser.add_argument('--predict', type=int, help='Number of characters to predict')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for prediction')
    parser.add_argument('--interactive', action='store_true', help='Run interactive mode')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not Path(args.model).exists():
        print(f"Error: Model file '{args.model}' not found")
        sys.exit(1)
    
    # Create tester
    tester = MarkovTester(args.model)
    
    if args.interactive or (not args.text):
        # Run interactive mode
        interactive_mode(tester)
    
    elif args.text:
        print(f"\nInput text: '{args.text}'")
        
        if args.predict:
            # Predict multiple characters
            result = tester.predict_multiple_chars(
                args.text, 
                num_chars=args.predict, 
                temperature=args.temperature
            )
            if isinstance(result, str):
                print(result)
            else:
                print(f"Generated: '{result['generated_text']}'")
                print(f"Continuation: '{result['continuation_only']}'")
        else:
            # Predict single character
            result = tester.predict_single_char(args.text)
            if isinstance(result, str):
                print(result)
            else:
                print(f"\nNext character prediction:")
                print(f"State: '{result['state']}'")
                print(f"Best prediction: '{result['best_prediction']}'")
                print(f"\nTop 5 predictions:")
                for char, prob in result['top_5_predictions']:
                    display_char = repr(char)[1:-1]
                    print(f"  '{display_char}': {prob:.3f} ({prob*100:.1f}%)")


if __name__ == "__main__":
    main()