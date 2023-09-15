import os
import torch

def expand_with_zeros(original_tensor,desired_size=20):
    batch_size = original_tensor.shape[0]
    # Define the desired size for the expanded tensor
    expanded_size = (desired_size, desired_size)

    # Create a new tensor of the desired size filled with zeros
    expanded_tensor = torch.zeros(batch_size, *expanded_size)

    # Calculate the padding needed on each side
    padding_top = (expanded_size[0] - original_tensor.shape[1]) // 2
    padding_left = (expanded_size[1] - original_tensor.shape[2]) // 2

    # Copy each original tensor into the center of the corresponding expanded tensor
    expanded_tensor[:, padding_top:padding_top + original_tensor.shape[1],
    padding_left:padding_left + original_tensor.shape[2]] = original_tensor

    return expanded_tensor



