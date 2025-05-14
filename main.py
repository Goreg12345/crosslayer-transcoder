import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)

if cuda_available:
    # Number of GPUs
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs:", num_gpus)

    # List each device’s name
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {name}")
else:
    print("No CUDA devices found")
