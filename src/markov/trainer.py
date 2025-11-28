#!/usr/bin/env python3
"""
Trainer for Character-level Markov Model using Common Crawl Dataset
Author: Assistant
Description: Train a Markov model on Common Crawl data (via C4 dataset)
"""

import os
import sys
import time
import pickle
import argparse
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

# Check for required packages
try:
    from datasets import load_dataset, IterableDataset
    from tqdm import tqdm
except ImportError:
    print("Required packages not installed. Please run:")
    print("pip install datasets tqdm")
    sys.exit(1)

# Import the MarkovModel from markov_model.py
try:
    from markov_model import MarkovModel
except ImportError:
    print("Error: markov_model.py not found in current directory")
    print("Please ensure markov_model.py is in the same directory as trainer.py")
    sys.exit(1)


class CommonCrawlTrainer:
    """Trainer for Markov Model using Common Crawl/C4 dataset"""
    
    def __init__(self, model_order=3, max_chars=1000000, checkpoint_dir="checkpoints"):
        """
        Initialize trainer
        
        Args:
            model_order: Order of the Markov model
            max_chars: Maximum characters to train on (for memory management)
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = MarkovModel(order=model_order)
        self.max_chars = max_chars
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training statistics
        self.total_chars_processed = 0
        self.total_docs_processed = 0
        self.start_time = None
    
    def load_dataset_streaming(self, dataset_name="c4", subset="en", split="train"):
        """
        Load Common Crawl dataset in streaming mode to handle large size
        
        Args:
            dataset_name: Name of the dataset ('c4' is Common Crawl cleaned)
            subset: Language subset (default 'en' for English)
            split: Dataset split to use
        
        Returns:
            Streaming dataset iterator
        """
        print(f"\nLoading {dataset_name} dataset (streaming mode)...")
        print("Note: First access may take time to download dataset info")
        
        try:
            # C4 is a cleaned version of Common Crawl
            # Using streaming=True to handle the massive size
            dataset = load_dataset(
                dataset_name,
                subset,
                split=split,
                streaming=True,
                trust_remote_code=True
            )
            return dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("\nTrying alternative: allenai/c4")
            # Alternative C4 location
            dataset = load_dataset(
                "allenai/c4",
                subset,
                split=split,
                streaming=True,
                trust_remote_code=True
            )
            return dataset
    
    def process_text_batch(self, text_batch, min_length=100):
        """
        Process and clean a batch of text
        
        Args:
            text_batch: List of text strings
            min_length: Minimum text length to include
        
        Returns:
            Concatenated cleaned text
        """
        processed_texts = []
        
        for text in text_batch:
            if len(text) >= min_length:
                # Basic cleaning
                text = text.strip()
                # Replace multiple spaces/newlines with single ones
                text = ' '.join(text.split())
                # Add text with a separator
                processed_texts.append(text)
        
        # Join texts with newlines to preserve document boundaries
        return '\n'.join(processed_texts)
    
    def train_incremental(self, dataset, num_samples=10000, batch_size=100, 
                         save_interval=1000, verbose=True):
        """
        Train model incrementally on streaming dataset
        
        Args:
            dataset: Streaming dataset
            num_samples: Number of samples to train on
            batch_size: Number of samples to process at once
            save_interval: Save checkpoint every N samples
            verbose: Print progress information
        """
        self.start_time = time.time()
        
        print(f"\nStarting training:")
        print(f"  - Model order: {self.model.order}")
        print(f"  - Target samples: {num_samples:,}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Max chars per training: {self.max_chars:,}")
        print(f"  - Checkpoint interval: {save_interval:,} samples")
        
        text_buffer = []
        buffer_size = 0
        
        # Progress bar
        pbar = tqdm(total=num_samples, desc="Processing samples", unit="docs")
        
        try:
            for i, example in enumerate(dataset.take(num_samples)):
                # Get text from the example (C4 uses 'text' field)
                text = example.get('text', '') if isinstance(example, dict) else str(example)
                
                if text:
                    text_buffer.append(text)
                    buffer_size += len(text)
                    self.total_docs_processed += 1
                
                # Process batch when buffer is full or at specified intervals
                if (i + 1) % batch_size == 0 or buffer_size >= self.max_chars:
                    if text_buffer:
                        # Process and train on batch
                        batch_text = self.process_text_batch(text_buffer)
                        if len(batch_text) > self.model.order:
                            self._train_on_text(batch_text, verbose=False)
                            self.total_chars_processed += len(batch_text)
                        
                        # Clear buffer
                        text_buffer = []
                        buffer_size = 0
                
                # Save checkpoint
                if (i + 1) % save_interval == 0:
                    self.save_checkpoint(f"checkpoint_{i+1}_samples.pkl")
                    if verbose:
                        self._print_statistics()
                
                # Update progress
                pbar.update(1)
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
        except Exception as e:
            print(f"\n\nError during training: {e}")
        finally:
            pbar.close()
            
            # Process remaining buffer
            if text_buffer:
                batch_text = self.process_text_batch(text_buffer)
                if len(batch_text) > self.model.order:
                    self._train_on_text(batch_text, verbose=False)
                    self.total_chars_processed += len(batch_text)
            
            # Final statistics
            self._print_statistics()
    
    def _train_on_text(self, text, verbose=False):
        """
        Train model on a text string, updating existing transitions
        
        Args:
            text: Text to train on
            verbose: Print training info
        """
        # Build transitions incrementally
        for i in range(len(text) - self.model.order):
            current_state = text[i:i + self.model.order]
            next_char = text[i + self.model.order]
            
            self.model.transitions[current_state][next_char] += 1
            self.model.states.add(current_state)
            self.model.char_vocab.add(next_char)
        
        # Add final state
        self.model.states.add(text[-self.model.order:])
        
        # Add all characters to vocabulary
        for char in text:
            self.model.char_vocab.add(char)
        
        if verbose:
            print(f"Trained on {len(text):,} characters")
    
    def _print_statistics(self):
        """Print training statistics"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        chars_per_sec = self.total_chars_processed / elapsed if elapsed > 0 else 0
        
        print(f"\n--- Training Statistics ---")
        print(f"Documents processed: {self.total_docs_processed:,}")
        print(f"Characters processed: {self.total_chars_processed:,}")
        print(f"Unique states: {len(self.model.states):,}")
        print(f"Unique transitions: {sum(len(v) for v in self.model.transitions.values()):,}")
        print(f"Vocabulary size: {len(self.model.char_vocab)}")
        print(f"Processing speed: {chars_per_sec:.0f} chars/sec")
        print(f"Time elapsed: {elapsed:.1f} seconds")
    
    def save_checkpoint(self, filename=None):
        """Save model checkpoint"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_checkpoint_{timestamp}.pkl"
        
        filepath = self.checkpoint_dir / filename
        
        # Save model
        self.model.save_model(filepath)
        
        # Save training metadata
        metadata_file = self.checkpoint_dir / f"{filename}.meta"
        with open(metadata_file, 'w') as f:
            f.write(f"total_chars: {self.total_chars_processed}\n")
            f.write(f"total_docs: {self.total_docs_processed}\n")
            f.write(f"unique_states: {len(self.model.states)}\n")
            f.write(f"vocabulary_size: {len(self.model.char_vocab)}\n")
        
        print(f"Checkpoint saved: {filepath}")
    
    def generate_samples(self, num_samples=5, length=200, temperature=0.8):
        """Generate sample texts to evaluate model quality"""
        print(f"\n--- Generated Samples (temp={temperature}) ---")
        
        # Common starting phrases
        seed_texts = [
            "The ", "In the ", "Once upon", "There was", "It was",
            "When ", "After ", "Before ", "During ", "While "
        ]
        
        for i in range(min(num_samples, len(seed_texts))):
            seed = random.choice(seed_texts) if i >= len(seed_texts) else seed_texts[i]
            
            try:
                generated = self.model.generate_text(
                    length=length,
                    seed_text=seed,
                    temperature=temperature
                )
                print(f"\nSample {i+1} (seed: '{seed}'):")
                print(f"{generated[:200]}...")
            except Exception as e:
                print(f"Error generating sample {i+1}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Train Markov Model on Common Crawl Dataset')
    
    # Model parameters
    parser.add_argument('--order', type=int, default=3, 
                      help='Markov model order (default: 3)')
    parser.add_argument('--max-chars', type=int, default=1000000,
                      help='Max characters to buffer in memory (default: 1M)')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='c4',
                      help='Dataset name (default: c4)')
    parser.add_argument('--language', type=str, default='en',
                      help='Language subset (default: en)')
    parser.add_argument('--num-samples', type=int, default=10000,
                      help='Number of documents to train on (default: 10000)')
    parser.add_argument('--batch-size', type=int, default=100,
                      help='Batch size for processing (default: 100)')
    
    # Training parameters
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                      help='Directory for checkpoints (default: checkpoints)')
    parser.add_argument('--save-interval', type=int, default=1000,
                      help='Save checkpoint every N samples (default: 1000)')
    parser.add_argument('--final-model', type=str, default='final_model.pkl',
                      help='Final model filename (default: final_model.pkl)')
    
    # Generation parameters
    parser.add_argument('--generate', action='store_true',
                      help='Generate sample texts after training')
    parser.add_argument('--num-generations', type=int, default=5,
                      help='Number of sample texts to generate (default: 5)')
    parser.add_argument('--generation-length', type=int, default=200,
                      help='Length of generated samples (default: 200)')
    parser.add_argument('--temperature', type=float, default=0.8,
                      help='Generation temperature (default: 0.8)')
    
    # Resume training
    parser.add_argument('--resume', type=str,
                      help='Resume from checkpoint file')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = CommonCrawlTrainer(
        model_order=args.order,
        max_chars=args.max_chars,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.model.load_model(args.resume)
        # Note: This doesn't restore training statistics, just the model
    
    # Load dataset
    try:
        dataset = trainer.load_dataset_streaming(
            dataset_name=args.dataset,
            subset=args.language
        )
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("\nYou may need to install additional dependencies:")
        print("pip install datasets apache-beam")
        sys.exit(1)
    
    # Train model
    print("\nStarting training on Common Crawl data...")
    trainer.train_incremental(
        dataset=dataset,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        save_interval=args.save_interval,
        verbose=True
    )
    
    # Save final model
    print(f"\nSaving final model to {args.final_model}")
    trainer.model.save_model(args.final_model)
    
    # Generate samples if requested
    if args.generate:
        trainer.generate_samples(
            num_samples=args.num_generations,
            length=args.generation_length,
            temperature=args.temperature
        )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()