import torch

def spins_to_bool(x):
    """Convert a 1 and -1 vector to a boolean vector"""
    return (x == 1).type(torch.bool)

def bool_to_spins(x):
    """Convert a boolean vector to a 1 and -1 vector"""
    return (2*x.type(torch.int8) - 1).type(torch.float32)

def get_bool_flips(x):
    """Get all spin flips of a binary vector, one dimension at a time"""
    flips = x.unsqueeze(0) ^ torch.eye(x.shape[0], dtype=x.dtype)
    return flips

def get_spin_flips(x):
  return bool_to_spins(get_bool_flips(spins_to_bool(x)))

def flip_and_copy_bool(X):
    """
    Here we flip:
    Args:
        X torch.Tensor(number_of_paths,number_of_spins): sample
    Returns:
        X_copy,X_flipped torch.Tensor(number_of_paths*number_of_spins, number_of_spins):
    """
    number_of_spins = X.shape[1]
    number_of_paths = X.shape[0]

    flip_mask = torch.eye(number_of_spins)[None ,: ,:].repeat_interleave(number_of_paths ,0).bool()
    X_flipped = X[: ,None ,:].bool() ^ flip_mask.bool()
    X_flipped = X_flipped.reshape(number_of_paths *number_of_spins ,number_of_spins)
    X_copy = X.repeat_interleave(number_of_spins ,0)

    return X_copy ,X_flipped

def copy_and_flip_spins(X_spins):
    #TEST
    X_bool = spins_to_bool(X_spins)
    X_copy_bool ,X_flipped_bool = flip_and_copy_bool(X_bool)
    X_copy_spin,X_flipped_spin = bool_to_spins(X_copy_bool), bool_to_spins(X_flipped_bool)
    return X_copy_spin, X_flipped_spin
