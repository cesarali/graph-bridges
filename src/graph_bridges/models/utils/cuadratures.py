import torch
import numpy as np

def gauss_legendre_weights_and_nodes(n, t0, t1):
    """
    Return the weights and nodes for n-point Gauss-Legendre quadrature
    scaled to the interval [t0, t1].
    """
    # Get nodes and weights for the standard interval [-1, 1]
    nodes, weights = np.polynomial.legendre.leggauss(n)

    # Scale nodes and weights to [t0, t1]
    nodes = 0.5 * (t1 - t0) * nodes + 0.5 * (t1 + t0)
    weights *= 0.5 * (t1 - t0)

    return torch.tensor(nodes, dtype=torch.float32), torch.tensor(weights, dtype=torch.float32)


def compute_w_t_with_quadrature(beta_function, S, t0, t1, n_points):
    """
    Compute w_t using Gaussian quadrature.
    """
    nodes, weights = gauss_legendre_weights_and_nodes(n_points, t0, t1)
    integral_value = torch.sum(beta_function(nodes) * weights)

    w_t = torch.exp(-S * integral_value)

    return w_t