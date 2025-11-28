# Character-level Markov Model for Text Generation

A Python implementation of a character-level Markov model for text prediction and generation, with support for training on large datasets like Common Crawl.

## Files Overview

- **`markov_model.py`** - Core Markov model implementation with training, prediction, and generation capabilities
- **`trainer.py`** - Trainer for large-scale training on Common Crawl dataset with incremental learning
- **`tester.py`** - Interactive testing utilities for model evaluation and text prediction

## Requirements
```bash
pip install datasets tqdm
```

For Common Crawl training (optional):
```bash
pip install apache-beam
```

## Quick Start

### 1. Training a Model

#### Option A: Train on a text file
```bash
# Train on a local text file
python markov_model.py --train data.txt --order 3 --save model.pkl

# Train and generate sample text
python markov_model.py --train data.txt --order 4 --generate 200 --save model.pkl
```

#### Option B: Train on Common Crawl dataset
```bash
# Train on 10,000 Common Crawl documents
python trainer.py --num-samples 10000 --order 3 --final-model cc_model.pkl

# Train with custom parameters
python trainer.py \
    --num-samples 50000 \
    --order 4 \
    --batch-size 100 \
    --save-interval 5000 \
    --checkpoint-dir checkpoints \
    --final-model final_model.pkl \
    --generate \
    --num-generations 5
```

### 2. Using a Trained Model

#### Generate text with markov_model.py
```bash
# Load model and generate text
python markov_model.py --load model.pkl --generate 500 --seed "The " --temperature 0.8

# Evaluate model perplexity
python markov_model.py --load model.pkl --evaluate test.txt
```

#### Interactive testing with tester.py
```bash
# Predict next characters
python tester.py model.pkl predict "Hello wor"

# Autocomplete text
python tester.py model.pkl complete "Once upon a " --length 100 --randomness 0.7

# Analyze model's knowledge of a text
python tester.py model.pkl analyze "The quick brown fox" --show-all

# Interactive mode
python tester.py model.pkl interactive
```

## Detailed Usage

### markov_model.py

Main script for basic Markov model operations.

**Arguments:**
- `--train FILE` - Train model on text file
- `--order N` - Set model order (default: 3)
- `--generate N` - Generate N characters
- `--seed TEXT` - Seed text for generation
- `--temperature FLOAT` - Generation randomness (0.5=less random, 2.0=more random)
- `--save FILE` - Save model to file
- `--load FILE` - Load model from file
- `--evaluate FILE` - Calculate perplexity on test file

**Examples:**
```bash
# Train order-5 model
python markov_model.py --train corpus.txt --order 5 --save model_o5.pkl

# Generate creative text (high temperature)
python markov_model.py --load model.pkl --generate 300 --temperature 1.5

# Generate conservative text (low temperature)
python markov_model.py --load model.pkl --generate 300 --temperature 0.5
```

### trainer.py

Specialized trainer for large-scale training on Common Crawl data with streaming and checkpointing.

**Key Arguments:**
- `--order N` - Markov model order
- `--num-samples N` - Number of documents to train on
- `--batch-size N` - Documents per batch
- `--save-interval N` - Checkpoint frequency
- `--checkpoint-dir DIR` - Checkpoint directory
- `--final-model FILE` - Final model filename
- `--resume FILE` - Resume from checkpoint
- `--generate` - Generate samples after training

**Examples:**
```bash
# Basic training
python trainer.py --num-samples 5000

# Large-scale training with checkpoints
python trainer.py \
    --num-samples 100000 \
    --order 4 \
    --batch-size 200 \
    --save-interval 10000 \
    --checkpoint-dir models/checkpoints \
    --final-model models/cc_large.pkl

# Resume interrupted training
python trainer.py \
    --resume checkpoints/checkpoint_50000_samples.pkl \
    --num-samples 100000 \
    --final-model models/cc_resumed.pkl
```

### tester.py

Interactive testing and analysis utilities.

**Commands:**

1. **predict** - Show top character predictions
```bash
   python tester.py model.pkl predict "Hello wo" --top-k 5
```

2. **complete** - Autocomplete text
```bash
   python tester.py model.pkl complete "The weather is " --length 50 --randomness 0.8
```

3. **analyze** - Analyze model's knowledge
```bash
   python tester.py model.pkl analyze "Sample text here" --show-all
```

4. **interactive** - Interactive testing mode
```bash
   python tester.py model.pkl interactive
   # Then use commands:
   # > predict The quick
   # > complete Once upon
   # > analyze Hello world
   # > quit
```

## Training Workflow Example
```bash
# Step 1: Initial training
python trainer.py \
    --num-samples 20000 \
    --order 3 \
    --checkpoint-dir training_run1 \
    --final-model model_v1.pkl

# Step 2: Test the model
python tester.py model_v1.pkl predict "The "
python tester.py model_v1.pkl complete "In the beginning" --length 100

# Step 3: Evaluate quality
python markov_model.py --load model_v1.pkl --evaluate test_data.txt

# Step 4: Generate samples
python markov_model.py \
    --load model_v1.pkl \
    --generate 500 \
    --seed "Once upon a time" \
    --temperature 0.8
```

## Model Parameters

### Order
- **Order 2-3**: Fast, less context, more creative/chaotic
- **Order 4-5**: Good balance of coherence and creativity
- **Order 6+**: More coherent but requires much more training data

### Temperature
- **0.5**: Conservative, predictable text
- **0.8-1.0**: Balanced creativity (recommended)
- **1.5-2.0**: Very creative, may be less coherent

### Training Size
- **1,000 documents**: Quick testing
- **10,000 documents**: Reasonable quality
- **50,000+ documents**: High quality generation

## Tips

1. **Memory Management**: Use `--max-chars` in trainer.py to control memory usage
2. **Checkpointing**: Always use checkpoints for long training runs
3. **Evaluation**: Use perplexity to compare models (lower is better)
4. **Seed Text**: Provide good seed text that exists in training data for better results
5. **Temperature Tuning**: Start with 0.8 and adjust based on output quality

## Output Examples
```bash
# Prediction output
$ python tester.py model.pkl predict "The sun "
✓ Model loaded: order=3, states=125,432

State: 'sun' → next character predictions:
  1. ' ' : 0.421 (42.1%)
  2. 's' : 0.234 (23.4%)
  3. ',' : 0.089 (8.9%)
  4. '.' : 0.076 (7.6%)
  5. 'n' : 0.065 (6.5%)

# Completion output
$ python tester.py model.pkl complete "The weather" --length 50
Original:  The weather
Completed: The weather was beautiful that morning as the sun rose
Added (42 chars): was beautiful that morning as the sun rose
```