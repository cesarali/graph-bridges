import os
import torch
import numpy as np
from torchvision import transforms

# Create a custom transformation class

class ToUpperDiagonalIndicesTransform:

    def __call__(self, tensor):
        if  len(tensor.shape) == 3:
            batch_size = tensor.shape[0]
            # Get the upper diagonal entries without zero-padding with the batch as the first dimension
            upper_diagonal_entries = tensor.masked_select(torch.triu(torch.ones_like(tensor), diagonal=1).bool())
            upper_diagonal_entries = upper_diagonal_entries.reshape(batch_size, -1)
            return upper_diagonal_entries
        else:
            raise Exception("Wrong Tensor Shape in Transform")

class FromUpperDiagonalTransform:

    def __call__(self, upper_diagonal_tensor):
        assert len(upper_diagonal_tensor.shape) == 2
        number_of_upper_entries = upper_diagonal_tensor.shape[1]
        batch_size = upper_diagonal_tensor.shape[0]

        matrix_size = int(.5 * (1 + np.sqrt(1 + 8 * number_of_upper_entries)))

        # Create a zero-filled tensor to hold the full matrices
        full_matrices = torch.zeros(batch_size, matrix_size, matrix_size, device=upper_diagonal_tensor.device)

        # Get the indices for the upper diagonal part of the matrices
        upper_tri_indices = torch.triu_indices(matrix_size, matrix_size, offset=1, device=upper_diagonal_tensor.device)

        # Fill the upper diagonal part of the matrices
        full_matrices[:, upper_tri_indices[0], upper_tri_indices[1]] = upper_diagonal_tensor

        # Transpose and fill the lower diagonal part to make the matrices symmetric
        full_matrices = full_matrices + full_matrices.transpose(1, 2)

        return full_matrices

class UnsqueezeTensorTransform:

    def __init__(self,axis=0):
        self.axis = axis
    def __call__(self, tensor:torch.Tensor):
        return tensor.unsqueeze(self.axis)

class BinaryTensorToSpinsTransform:

    def __call__(self,binary_tensor):
        spins = (-1.) ** (binary_tensor + 1)
        return spins


SqueezeTransform = transforms.Lambda(lambda x: x.squeeze())
FlattenTransform = transforms.Lambda(lambda x: x.reshape(x.shape[0], -1))
UnFlattenTransform = transforms.Lambda(lambda x: x.reshape(x.shape[0],
                                                           int(np.sqrt(x.shape[1])),
                                                           int(np.sqrt(x.shape[1]))))
class SpinsToBinaryTensor:
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be transformed.
        Returns:
            Tensor: Transformed tensor.
        """
        # Create a copy of the input tensor to avoid modifying the original tensor
        transformed_tensor = tensor.clone()
        # Replace -1. with 0. in the tensor
        transformed_tensor[transformed_tensor == -1.] = 0.

        return transformed_tensor

BinaryTensorToSpinsTransform = transforms.Lambda(lambda binary_tensor: (-1.) ** (binary_tensor + 1))