"""
Basic Tensor Operations in PyTorch
This script demonstrates fundamental tensor operations.
"""

import torch
import numpy as np

def main():
    print("=" * 50)
    print("PyTorch Basic Tensor Operations")
    print("=" * 50)
    
    # 1. Creating tensors
    print("\n1. Creating Tensors:")
    print("-" * 30)
    
    # From Python list
    tensor1 = torch.tensor([1, 2, 3, 4, 5])
    print(f"From list: {tensor1}")
    
    # From numpy array
    np_array = np.array([1, 2, 3, 4, 5])
    tensor2 = torch.from_numpy(np_array)
    print(f"From numpy: {tensor2}")
    
    # Zeros and ones
    zeros = torch.zeros(3, 3)
    ones = torch.ones(2, 4)
    print(f"\nZeros (3x3):\n{zeros}")
    print(f"\nOnes (2x4):\n{ones}")
    
    # Random tensor
    random_tensor = torch.randn(2, 3)
    print(f"\nRandom tensor (2x3):\n{random_tensor}")
    
    # 2. Tensor operations
    print("\n\n2. Tensor Operations:")
    print("-" * 30)
    
    a = torch.tensor([1, 2, 3], dtype=torch.float32)
    b = torch.tensor([4, 5, 6], dtype=torch.float32)
    
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    print(f"a @ b (dot product) = {a @ b}")
    
    # 3. Reshaping
    print("\n\n3. Reshaping Tensors:")
    print("-" * 30)
    
    original = torch.arange(12)
    print(f"Original (1D): {original}")
    reshaped = original.reshape(3, 4)
    print(f"Reshaped (3x4):\n{reshaped}")
    
    # 4. GPU support
    print("\n\n4. GPU Support:")
    print("-" * 30)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        tensor_gpu = torch.tensor([1, 2, 3]).cuda()
        print(f"Tensor on GPU: {tensor_gpu.device}")
    else:
        print("CUDA not available, using CPU")
    
    print("\n" + "=" * 50)
    print("Basic operations completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()

