import torch
from safetensors.torch import save_file, safe_open

# Open the safetensors file
model = safe_open('F:/sd3_test/sd3_medium.safetensors', framework='pt')  # Updated the backslashes to forward slashes for compatibility

# Retrieve keys and create a dictionary of tensors
keys = model.keys()
dic = {key: model.get_tensor(key) for key in keys}

# Part of the key to filter by
parts = ['diffusion_model']
count = 0

# Loop through the keys
for k in keys:
    # Check if all parts are in the key
    if all(part in k for part in parts):
        v = dic[k]  # Get the tensor
        print(f'{k}: {v.std()}')  # Corrected the print statement to use 'v'
        
        # Add noise to the tensor
        noise = torch.normal(torch.zeros_like(v) * v.mean(), torch.ones_like(v) * v.std() * 0.02)
        dic[k] += noise
        
        count += 1

# Print the count of modified tensors
print(count)

# Save the modified tensors to a new file
save_file(dic, 'F:/sd3_test/sd3_medium_perturbed_marduk191-3.safetensors', metadata=model.metadata())
