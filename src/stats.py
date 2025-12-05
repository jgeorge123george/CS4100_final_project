import torch

# Load the checkpoint
checkpoint = torch.load('model_state_emnist_byclass.pt', map_location='cpu')

# Get the model state dict
model_state_dict = checkpoint['model_state_dict']

# Count parameters
total_params = sum(p.numel() for p in model_state_dict.values() if isinstance(p, torch.Tensor))

print(f"Total parameters: {total_params:,}")

# Show detailed breakdown
print(f"\nModel info:")
print(f"Number of classes: {checkpoint['num_classes']}")
print(f"Split: {checkpoint['split']}")

print(f"\nParameter breakdown:")
print("-" * 60)
for name, param in model_state_dict.items():
    if isinstance(param, torch.Tensor):
        print(f"{name:<30} {str(param.shape):<20} {param.numel():>10,} params")
print("-" * 60)
print(f"Total parameters: {total_params:,}")

# Calculate approximate model size
size_mb = total_params * 4 / 1024 / 1024  # assuming float32
print(f"Approximate model size: {size_mb:.2f} MB")