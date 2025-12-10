#!/usr/bin/env python3
"""
Trainer for Character-level Markov Model using Common Crawl Dataset
Memory-optimized version with optional pruning
"""

import os
import sys
import time
import pickle
import argparse
import random
import json
import gc
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

try:
    from datasets import load_dataset, IterableDataset
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Required packages not installed. Please run:")
    print("pip install datasets tqdm matplotlib numpy")
    sys.exit(1)

try:
    from markov_model import MarkovModel
except ImportError:
    print("Error: markov_model.py not found in current directory")
    sys.exit(1)


class RunningStats:
    """Memory-efficient running statistics (no storage of individual values)"""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # For Welford's algorithm
        self.min_val = float('inf')
        self.max_val = float('-inf')
    
    def update(self, x):
        self.n += 1
        self.min_val = min(self.min_val, x)
        self.max_val = max(self.max_val, x)
        # Welford's online algorithm for mean and variance
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
    
    @property
    def variance(self):
        return self.M2 / self.n if self.n > 1 else 0.0
    
    @property
    def std(self):
        return self.variance ** 0.5
    
    def to_dict(self):
        return {
            'count': self.n, 'mean': self.mean, 'std': self.std,
            'min': self.min_val if self.n > 0 else None,
            'max': self.max_val if self.n > 0 else None
        }


class TrainingStats:
    """Track training statistics with bounded memory"""
    def __init__(self, max_points=2000):
        self.max_points = max_points
        self.timestamps = []
        self.chars_processed = []
        self.docs_processed = []
        self.unique_states = []
        self.vocab_sizes = []
        self.transition_counts = []
        self.chars_per_second = []
        self.memory_mb = []
        
    def record(self, timestamp, chars, docs, states, vocab, transitions, speed, memory=0):
        # Downsample if we have too many points
        if len(self.timestamps) >= self.max_points:
            self._downsample()
        
        self.timestamps.append(timestamp)
        self.chars_processed.append(chars)
        self.docs_processed.append(docs)
        self.unique_states.append(states)
        self.vocab_sizes.append(vocab)
        self.transition_counts.append(transitions)
        self.chars_per_second.append(speed)
        self.memory_mb.append(memory)
    
    def _downsample(self):
        """Keep every other point to stay within memory bounds"""
        self.timestamps = self.timestamps[::2]
        self.chars_processed = self.chars_processed[::2]
        self.docs_processed = self.docs_processed[::2]
        self.unique_states = self.unique_states[::2]
        self.vocab_sizes = self.vocab_sizes[::2]
        self.transition_counts = self.transition_counts[::2]
        self.chars_per_second = self.chars_per_second[::2]
        self.memory_mb = self.memory_mb[::2]
    
    def to_dict(self):
        return {
            'timestamps': self.timestamps,
            'chars_processed': self.chars_processed,
            'docs_processed': self.docs_processed,
            'unique_states': self.unique_states,
            'vocab_sizes': self.vocab_sizes,
            'transition_counts': self.transition_counts,
            'chars_per_second': self.chars_per_second,
            'memory_mb': self.memory_mb
        }
    
    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def get_memory_mb():
    """Get current process memory usage in MB"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0


class CommonCrawlTrainer:
    """Memory-optimized trainer for Markov Model"""
    
    def __init__(self, model_order=3, max_chars=1000000, checkpoint_dir="checkpoints",
                 prune_threshold=0, prune_interval=100000):
        self.model = MarkovModel(order=model_order)
        self.max_chars = max_chars
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Pruning settings
        self.prune_threshold = prune_threshold  # Remove transitions with count <= this
        self.prune_interval = prune_interval    # Prune every N documents
        self.total_pruned = 0
        
        self.total_chars_processed = 0
        self.total_docs_processed = 0
        self.start_time = None
        
        # Memory-efficient statistics
        self.stats = TrainingStats()
        self.doc_length_stats = RunningStats()  # Instead of storing all lengths
        self.char_frequencies = Counter()  # Bounded by vocab size, stays small
    
    def load_dataset_streaming(self, dataset_name="c4", subset="en", split="train"):
        print(f"\nLoading {dataset_name} dataset (streaming mode)...")
        try:
            dataset = load_dataset(
                dataset_name, subset, split=split,
                streaming=True, trust_remote_code=True
            )
            return dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("\nTrying alternative: allenai/c4")
            dataset = load_dataset(
                "allenai/c4", subset, split=split,
                streaming=True, trust_remote_code=True
            )
            return dataset
    
    def process_text_batch(self, text_batch, min_length=100):
        processed_texts = []
        for text in text_batch:
            if len(text) >= min_length:
                text = text.strip()
                text = ' '.join(text.split())
                processed_texts.append(text)
                self.doc_length_stats.update(len(text))  # Running stats, no storage
                self.char_frequencies.update(text)
        return '\n'.join(processed_texts)
    
    def prune_rare_transitions(self, threshold=1):
        """Remove transitions that have been seen <= threshold times"""
        if threshold <= 0:
            return 0
        
        pruned_count = 0
        states_to_remove = []
        
        for state, transitions in self.model.transitions.items():
            # Find rare transitions
            rare_chars = [char for char, count in transitions.items() if count <= threshold]
            for char in rare_chars:
                del transitions[char]
                pruned_count += 1
            
            # Mark empty states for removal
            if not transitions:
                states_to_remove.append(state)
        
        # Remove empty states
        for state in states_to_remove:
            del self.model.transitions[state]
            self.model.states.discard(state)
        
        return pruned_count
    
    def train_incremental(self, dataset, num_samples=10000, batch_size=100, 
                         save_interval=1000, stats_interval=100, verbose=True):
        self.start_time = time.time()
        
        print(f"\nStarting training:")
        print(f"  - Model order: {self.model.order}")
        print(f"  - Target samples: {num_samples:,}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Checkpoint interval: {save_interval:,} samples")
        print(f"  - Stats interval: {stats_interval:,} samples")
        if self.prune_threshold > 0:
            print(f"  - Pruning: remove transitions with count <= {self.prune_threshold} every {self.prune_interval:,} docs")
        
        text_buffer = []
        buffer_size = 0
        pbar = tqdm(total=num_samples, desc="Processing samples", unit="docs")
        
        try:
            for i, example in enumerate(dataset.take(num_samples)):
                text = example.get('text', '') if isinstance(example, dict) else str(example)
                
                if text:
                    text_buffer.append(text)
                    buffer_size += len(text)
                    self.total_docs_processed += 1
                
                if (i + 1) % batch_size == 0 or buffer_size >= self.max_chars:
                    if text_buffer:
                        batch_text = self.process_text_batch(text_buffer)
                        if len(batch_text) > self.model.order:
                            self._train_on_text(batch_text)
                            self.total_chars_processed += len(batch_text)
                        text_buffer = []
                        buffer_size = 0
                
                # Periodic pruning to control memory
                if self.prune_threshold > 0 and (i + 1) % self.prune_interval == 0:
                    pruned = self.prune_rare_transitions(self.prune_threshold)
                    self.total_pruned += pruned
                    if pruned > 0:
                        gc.collect()  # Force garbage collection
                        pbar.set_postfix({'pruned': f'{self.total_pruned:,}'})
                
                # Record statistics
                if (i + 1) % stats_interval == 0:
                    elapsed = time.time() - self.start_time
                    speed = self.total_chars_processed / elapsed if elapsed > 0 else 0
                    transitions = sum(len(v) for v in self.model.transitions.values())
                    self.stats.record(
                        timestamp=elapsed,
                        chars=self.total_chars_processed,
                        docs=self.total_docs_processed,
                        states=len(self.model.transitions),  # Use transitions keys directly
                        vocab=len(self.model.char_vocab),
                        transitions=transitions,
                        speed=speed,
                        memory=get_memory_mb()
                    )
                
                if (i + 1) % save_interval == 0:
                    self.save_checkpoint(f"checkpoint_{i+1}_samples.pkl")
                    if verbose:
                        self._print_statistics()
                
                pbar.update(1)
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
        except Exception as e:
            print(f"\n\nError during training: {e}")
            import traceback
            traceback.print_exc()
        finally:
            pbar.close()
            if text_buffer:
                batch_text = self.process_text_batch(text_buffer)
                if len(batch_text) > self.model.order:
                    self._train_on_text(batch_text)
                    self.total_chars_processed += len(batch_text)
            
            # Final prune if enabled
            if self.prune_threshold > 0:
                pruned = self.prune_rare_transitions(self.prune_threshold)
                self.total_pruned += pruned
            
            self._print_final_statistics()
    
    def _train_on_text(self, text):
        """Train on text - updates transitions directly without redundant state set"""
        order = self.model.order
        for i in range(len(text) - order):
            current_state = text[i:i + order]
            next_char = text[i + order]
            self.model.transitions[current_state][next_char] += 1
            self.model.char_vocab.add(next_char)
        
        # Add all characters to vocabulary
        for char in text:
            self.model.char_vocab.add(char)
    
    def _print_statistics(self):
        elapsed = time.time() - self.start_time if self.start_time else 0
        chars_per_sec = self.total_chars_processed / elapsed if elapsed > 0 else 0
        mem = get_memory_mb()
        
        print(f"\n--- Training Statistics ---")
        print(f"Documents processed: {self.total_docs_processed:,}")
        print(f"Characters processed: {self.total_chars_processed:,}")
        print(f"Unique states: {len(self.model.transitions):,}")
        print(f"Unique transitions: {sum(len(v) for v in self.model.transitions.values()):,}")
        print(f"Vocabulary size: {len(self.model.char_vocab)}")
        print(f"Processing speed: {chars_per_sec:.0f} chars/sec")
        print(f"Memory usage: {mem:.0f} MB")
        if self.total_pruned > 0:
            print(f"Transitions pruned: {self.total_pruned:,}")
    
    def _print_final_statistics(self):
        elapsed = time.time() - self.start_time if self.start_time else 0
        chars_per_sec = self.total_chars_processed / elapsed if elapsed > 0 else 0
        total_transitions = sum(len(v) for v in self.model.transitions.values())
        total_transition_weight = sum(
            sum(counts.values()) for counts in self.model.transitions.values()
        )
        mem = get_memory_mb()
        
        print("\n" + "="*60)
        print("           FINAL TRAINING STATISTICS")
        print("="*60)
        
        print("\n📊 DATASET STATISTICS:")
        print(f"  Documents processed:     {self.total_docs_processed:>15,}")
        print(f"  Characters processed:    {self.total_chars_processed:>15,}")
        doc_stats = self.doc_length_stats
        if doc_stats.n > 0:
            print(f"  Avg document length:     {doc_stats.mean:>15,.1f}")
            print(f"  Min document length:     {int(doc_stats.min_val):>15,}")
            print(f"  Max document length:     {int(doc_stats.max_val):>15,}")
            print(f"  Std dev doc length:      {doc_stats.std:>15,.1f}")
        
        print("\n🧠 MODEL STATISTICS:")
        print(f"  Model order:             {self.model.order:>15}")
        print(f"  Unique states:           {len(self.model.transitions):>15,}")
        print(f"  Unique transitions:      {total_transitions:>15,}")
        print(f"  Total transition weight: {total_transition_weight:>15,}")
        print(f"  Vocabulary size:         {len(self.model.char_vocab):>15}")
        print(f"  Avg transitions/state:   {total_transitions/max(1,len(self.model.transitions)):>15.2f}")
        
        print("\n⏱️  PERFORMANCE:")
        print(f"  Total time:              {elapsed:>15.1f} sec")
        print(f"  Processing speed:        {chars_per_sec:>15,.0f} chars/sec")
        print(f"  Docs per second:         {self.total_docs_processed/max(1,elapsed):>15.2f}")
        print(f"  Memory usage:            {mem:>15.0f} MB")
        
        if self.total_pruned > 0:
            print(f"\n✂️  PRUNING:")
            print(f"  Transitions pruned:      {self.total_pruned:>15,}")
        
        if self.char_frequencies:
            print("\n🔤 TOP 10 CHARACTERS:")
            total_chars = sum(self.char_frequencies.values())
            for char, count in self.char_frequencies.most_common(10):
                display = repr(char) if char in '\n\t\r ' else char
                pct = 100 * count / total_chars
                print(f"  {display:>8}: {count:>12,} ({pct:>5.2f}%)")
        
        if self.model.transitions:
            transition_counts = [sum(v.values()) for v in self.model.transitions.values()]
            print("\n🔗 TRANSITION DISTRIBUTION:")
            print(f"  Min count per state:     {min(transition_counts):>15,}")
            print(f"  Max count per state:     {max(transition_counts):>15,}")
            print(f"  Avg count per state:     {np.mean(transition_counts):>15,.2f}")
            print(f"  Std dev:                 {np.std(transition_counts):>15,.2f}")
        
        print("\n" + "="*60)
    
    def save_checkpoint(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_checkpoint_{timestamp}.pkl"
        
        filepath = self.checkpoint_dir / filename
        
        # Sync states set with transitions keys before saving
        self.model.states = set(self.model.transitions.keys())
        self.model.save_model(filepath)
        
        metadata_file = self.checkpoint_dir / f"{filename}.meta"
        with open(metadata_file, 'w') as f:
            f.write(f"total_chars: {self.total_chars_processed}\n")
            f.write(f"total_docs: {self.total_docs_processed}\n")
            f.write(f"unique_states: {len(self.model.transitions)}\n")
            f.write(f"vocabulary_size: {len(self.model.char_vocab)}\n")
            f.write(f"memory_mb: {get_memory_mb():.0f}\n")
        
        print(f"Checkpoint saved: {filepath}")
    
    def save_graphs(self, output_dir="graphs"):
        """Generate and save training visualization graphs"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig_size = (10, 6)
        
        print(f"\nGenerating graphs in '{output_dir}/'...")
        
        # 1. Training Progress Over Time (now includes memory)
        if self.stats.timestamps:
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            fig.suptitle('Training Progress Over Time', fontsize=14, fontweight='bold')
            
            axes[0, 0].plot(self.stats.timestamps, np.array(self.stats.chars_processed)/1e6, 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Time (seconds)')
            axes[0, 0].set_ylabel('Characters (millions)')
            axes[0, 0].set_title('Characters Processed')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(self.stats.timestamps, self.stats.docs_processed, 'g-', linewidth=2)
            axes[0, 1].set_xlabel('Time (seconds)')
            axes[0, 1].set_ylabel('Documents')
            axes[0, 1].set_title('Documents Processed')
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[0, 2].plot(self.stats.timestamps, np.array(self.stats.unique_states)/1e6, 'r-', linewidth=2)
            axes[0, 2].set_xlabel('Time (seconds)')
            axes[0, 2].set_ylabel('States (millions)')
            axes[0, 2].set_title('Unique States Growth')
            axes[0, 2].grid(True, alpha=0.3)
            
            axes[1, 0].plot(self.stats.timestamps, self.stats.chars_per_second, 'm-', linewidth=2)
            axes[1, 0].set_xlabel('Time (seconds)')
            axes[1, 0].set_ylabel('Chars/second')
            axes[1, 0].set_title('Processing Speed')
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(self.stats.timestamps, np.array(self.stats.transition_counts)/1e6, 'c-', linewidth=2)
            axes[1, 1].set_xlabel('Time (seconds)')
            axes[1, 1].set_ylabel('Transitions (millions)')
            axes[1, 1].set_title('Total Transitions')
            axes[1, 1].grid(True, alpha=0.3)
            
            if any(self.stats.memory_mb):
                axes[1, 2].plot(self.stats.timestamps, np.array(self.stats.memory_mb)/1024, 'orange', linewidth=2)
                axes[1, 2].set_xlabel('Time (seconds)')
                axes[1, 2].set_ylabel('Memory (GB)')
                axes[1, 2].set_title('Memory Usage')
                axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("  ✓ training_progress.png")
        
        # 2. Character Frequency Distribution
        if self.char_frequencies:
            fig, ax = plt.subplots(figsize=(12, 6))
            top_chars = self.char_frequencies.most_common(30)
            chars = [repr(c) if c in '\n\t\r ' else c for c, _ in top_chars]
            counts = [count for _, count in top_chars]
            
            ax.bar(range(len(chars)), counts, color='teal', edgecolor='white')
            ax.set_xticks(range(len(chars)))
            ax.set_xticklabels(chars, rotation=45, ha='right')
            ax.set_xlabel('Character')
            ax.set_ylabel('Frequency')
            ax.set_title('Top 30 Character Frequencies')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(output_dir / 'char_frequency.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("  ✓ char_frequency.png")
        
        # 3. Transition Count Distribution
        if self.model.transitions:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            trans_per_state = [len(v) for v in self.model.transitions.values()]
            axes[0].hist(trans_per_state, bins=50, color='coral', edgecolor='white', alpha=0.8)
            axes[0].set_xlabel('Number of Unique Transitions')
            axes[0].set_ylabel('Number of States')
            axes[0].set_title('Transitions per State Distribution')
            axes[0].axvline(np.mean(trans_per_state), color='red', linestyle='--',
                          label=f'Mean: {np.mean(trans_per_state):.1f}')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            weights_per_state = [sum(v.values()) for v in self.model.transitions.values()]
            axes[1].hist(weights_per_state, bins=50, color='mediumpurple', edgecolor='white', alpha=0.8)
            axes[1].set_xlabel('Total Transition Count')
            axes[1].set_ylabel('Number of States')
            axes[1].set_title('Transition Weight per State')
            axes[1].set_yscale('log')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'transition_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("  ✓ transition_distribution.png")
        
        # 4. Model Growth Summary
        if self.stats.timestamps and len(self.stats.timestamps) > 1:
            fig, ax = plt.subplots(figsize=fig_size)
            
            docs = np.array(self.stats.docs_processed)
            states = np.array(self.stats.unique_states)
            transitions = np.array(self.stats.transition_counts)
            
            ax.plot(docs, states/max(states.max(), 1), 'b-', label='States (normalized)', linewidth=2)
            ax.plot(docs, transitions/max(transitions.max(), 1), 'r-', label='Transitions (normalized)', linewidth=2)
            
            ax.set_xlabel('Documents Processed')
            ax.set_ylabel('Normalized Value')
            ax.set_title('Model Growth vs Documents Processed')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'model_growth.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("  ✓ model_growth.png")
        
        # 5. Memory vs States
        if self.stats.memory_mb and any(self.stats.memory_mb):
            fig, ax = plt.subplots(figsize=fig_size)
            ax.scatter(np.array(self.stats.unique_states)/1e6, 
                      np.array(self.stats.memory_mb)/1024, 
                      alpha=0.5, c='purple')
            ax.set_xlabel('Unique States (millions)')
            ax.set_ylabel('Memory (GB)')
            ax.set_title('Memory Usage vs Model Size')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'memory_vs_states.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("  ✓ memory_vs_states.png")
        
        self.stats.save(output_dir / 'training_stats.json')
        print("  ✓ training_stats.json")
        
        print(f"\n✅ All graphs saved to '{output_dir}/'")
    
    def generate_samples(self, num_samples=5, length=200, temperature=0.8):
        print(f"\n--- Generated Samples (temp={temperature}) ---")
        seed_texts = ["The ", "In the ", "Once upon", "There was", "It was",
                      "When ", "After ", "Before ", "During ", "While "]
        
        for i in range(min(num_samples, len(seed_texts))):
            seed = seed_texts[i] if i < len(seed_texts) else random.choice(seed_texts)
            try:
                generated = self.model.generate_text(length=length, seed_text=seed, temperature=temperature)
                print(f"\nSample {i+1} (seed: '{seed}'):")
                print(f"{generated[:200]}...")
            except Exception as e:
                print(f"Error generating sample {i+1}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Train Markov Model on Common Crawl Dataset')
    
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--max-chars', type=int, default=1000000)
    parser.add_argument('--dataset', type=str, default='c4')
    parser.add_argument('--language', type=str, default='en')
    parser.add_argument('--num-samples', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--save-interval', type=int, default=1000)
    parser.add_argument('--stats-interval', type=int, default=100)
    parser.add_argument('--final-model', type=str, default='final_model.pkl')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--num-generations', type=int, default=5)
    parser.add_argument('--generation-length', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--graphs-dir', type=str, default='graphs')
    parser.add_argument('--no-graphs', action='store_true')
    
    # New pruning arguments
    parser.add_argument('--prune-threshold', type=int, default=0,
                       help='Remove transitions seen <= this many times (0=disabled)')
    parser.add_argument('--prune-interval', type=int, default=100000,
                       help='Prune every N documents (default: 100000)')
    
    args = parser.parse_args()
    
    trainer = CommonCrawlTrainer(
        model_order=args.order,
        max_chars=args.max_chars,
        checkpoint_dir=args.checkpoint_dir,
        prune_threshold=args.prune_threshold,
        prune_interval=args.prune_interval
    )
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.model.load_model(args.resume)
    
    try:
        dataset = trainer.load_dataset_streaming(dataset_name=args.dataset, subset=args.language)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)
    
    print("\nStarting training on Common Crawl data...")
    trainer.train_incremental(
        dataset=dataset,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        save_interval=args.save_interval,
        stats_interval=args.stats_interval,
        verbose=True
    )
    
    # Sync states before final save
    trainer.model.states = set(trainer.model.transitions.keys())
    
    print(f"\nSaving final model to {args.final_model}")
    trainer.model.save_model(args.final_model)
    
    if not args.no_graphs:
        trainer.save_graphs(output_dir=args.graphs_dir)
    
    if args.generate:
        trainer.generate_samples(
            num_samples=args.num_generations,
            length=args.generation_length,
            temperature=args.temperature
        )
    
    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()