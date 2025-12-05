import os
import subprocess
from pathlib import Path
import re
import matplotlib.pyplot as plt
import json

# Define the checkpoints folder path
checkpoints_dir = "checkpoints"

# Get all .pt files in the checkpoints folder
pt_files = sorted(Path(checkpoints_dir).glob("*.pt"))

results = {}  # Store epoch -> result mapping

if not pt_files:
    print(f"No .pt files found in {checkpoints_dir}")
else:
    for pt_file in pt_files:
        print(f"\nEvaluating {pt_file.name}...")
        cmd = ["python", "src/evaluate_test.py", str(pt_file)]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Extract epoch number from filename
            match = re.search(r'epoch_(\d+)', pt_file.name)
            if match:
                epoch = int(match.group(1))
                # Parse the result (adjust based on your evaluate_test.py output)
                # This assumes the last line contains the accuracy/metric
                output_lines = result.stdout.strip().split('\n')
                metric_value = float(output_lines[-1]) if output_lines else 0
                results[epoch] = metric_value
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating {pt_file.name}: {e}")

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

