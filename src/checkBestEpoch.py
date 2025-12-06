import os
import subprocess
from pathlib import Path
import re
import matplotlib.pyplot as plt
import json
import sys

# Change to the project root directory to ensure proper imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
os.chdir(project_root)

# Define the checkpoints folder path
checkpoints_dir = "checkpoints"

# Get all .pt files in the checkpoints folder
pt_files = sorted(Path(checkpoints_dir).glob("*.pt")) if Path(checkpoints_dir).exists() else []

results = {}  # Store epoch -> result mapping

if not pt_files:
    print(f"No .pt files found in {checkpoints_dir}")
    print("Checking if checkpoints directory exists...")
    if not Path(checkpoints_dir).exists():
        print(f"Directory '{checkpoints_dir}' does not exist.")
        print("You need to train a model with checkpoints enabled first.")
        print("Try running: python src/trainer.py --epochs 10 --split balanced")
    else:
        print(f"Directory '{checkpoints_dir}' exists but contains no .pt files.")
else:
    for pt_file in pt_files:
        print(f"\nEvaluating {pt_file.name}...")
        
        # Extract split from filename to pass correct arguments to evaluate_test.py
        split_match = re.search(r'emnist_(\w+)_epoch', pt_file.name)
        split = split_match.group(1) if split_match else "balanced"
        
        # Pass split first, then model path as expected by evaluate_test.py
        cmd = ["python", "src/evaluate_test.py", split, str(pt_file)]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
            
            # Extract epoch number from filename
            match = re.search(r'epoch_(\d+)', pt_file.name)
            if match:
                epoch = int(match.group(1))
                # Parse the result - look for accuracy line
                output_lines = result.stdout.strip().split('\n')
                accuracy_value = None
                for line in output_lines:
                    if "Accuracy" in line and "%" in line:
                        # Extract percentage value
                        acc_match = re.search(r'(\d+\.?\d*)%', line)
                        if acc_match:
                            accuracy_value = float(acc_match.group(1))
                            break
                
                if accuracy_value is not None:
                    results[epoch] = accuracy_value
                    print(f"Epoch {epoch}: {accuracy_value:.2f}% accuracy")
                else:
                    print(f"Could not parse accuracy from output for epoch {epoch}")
            else:
                print(f"Could not extract epoch number from {pt_file.name}")
                
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating {pt_file.name}: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"Error output: {e.stderr}")
        except subprocess.TimeoutExpired:
            print(f"Timeout evaluating {pt_file.name} - taking too long")

print("\nAll evaluations complete.")

# Create graph if results exist
if results:
    epochs = sorted(results.keys())
    metrics = [results[e] for e in epochs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Result')
    plt.title('Epoch vs Training Result')
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    print("Graph saved as 'training_results.png'")
    plt.show()