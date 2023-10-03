from typing import Tuple
import numpy as np
import torch

def marginals_histograms_mse(all_marginal_histograms)->Tuple[np.array,np.array]:
    """
    simply calculates the mse from the marginal graph histograms

    Returns
    -------
    mse_1,mse_0
    """
    marginal_0, marginal_generated_0, marginal_1, marginal_noising_1 = all_marginal_histograms
    if isinstance(marginal_0,torch.Tensor):
        marginal_0 = marginal_0.numpy()
    if isinstance(marginal_generated_0,torch.Tensor):
        marginal_generated_0 = marginal_generated_0.numpy()
    if isinstance(marginal_1,torch.Tensor):
        marginal_1 = marginal_1.numpy()
    if isinstance(marginal_noising_1,torch.Tensor):
        marginal_noising_1 = marginal_noising_1.numpy()

    mse_1 = np.mean((marginal_1 - marginal_noising_1)**2.)
    mse_0 = np.mean((marginal_0 - marginal_generated_0)**2.)

    return mse_1,mse_0