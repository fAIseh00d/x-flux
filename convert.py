import torch
from safetensors.torch import save_file

# Specify the input `.bin` file and output `.safetensors` file
input_file = "controlnet.bin"  # Replace with your `.bin` file path
output_file = "flux-densepose-controlnet.safetensors"

# Choose the desired precision
precision = torch.bfloat16  # Use `torch.float16` for FP16 or `torch.float8_e4m3` for FP8 (if supported)

# Load the PyTorch model
print("Loading PyTorch .bin file...")
state_dict = torch.load(input_file, map_location="cpu", weights_only=True)

# Ensure the data is a state_dict (dictionary of tensors)
if not isinstance(state_dict, dict):
    raise ValueError("The loaded file is not a valid state_dict.")

# Cast tensors to the desired precision
print(f"Casting tensors to {precision}...")
converted_state_dict = {k: v.to(precision) for k, v in state_dict.items()}

# Save to safetensors
print("Saving as .safetensors file...")
save_file(converted_state_dict, output_file)

print(f"Conversion complete! Saved to {output_file}")