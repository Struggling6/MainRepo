import torch
from config import TimesNetConfig

# Test model initialization
config = TimesNetConfig(
    num_layers=3,
    d_model=128,
    top_k=3,
    d_ffn=256,
    n_kernels=6,
    dropout=0.3,
    num_classes=1,
)

print("Building model...")
model = config.build(input_dim=46, context_length=168)
print(f"Model: {model}")

print("Testing forward pass...")
x = torch.randn(2, 168, 46)  # (batch, seq_length, features) - CHANGED!
print(f"Input shape: {x.shape}")

try:
    output = model(x)
    print(f"Output shape: {output.shape}")
    print("✓ Model works!")
except Exception as e:
    print(f"✗ Error: {e}")