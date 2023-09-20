import torch
from pprint import pprint
from torch.linalg import inv
from matplotlib import pyplot as plt

from graph_bridges.models.temporal_networks.embedding_utils import transformer_timestep_embedding

def get_marginal_spin_probability(paths):
    paths_count = torch.clone(paths)
    paths_count[torch.where(paths == -1.)] = 0.
    marginal_spin_probability = paths_count.sum(axis=0)/paths.shape[0]
    return marginal_spin_probability

def kl_between_spin_marginals(marginal_spin_probability_forward,marginal_spin_probability_backward):
    kl = marginal_spin_probability_forward*(torch.log(marginal_spin_probability_forward) - torch.log(marginal_spin_probability_backward))
    kl = kl.sum(axis=1)
    return kl

def log_likelihood_of_path(parametrized_ising_hamiltonian,paths,plot=True):
    """
    :param parametrized_ising_hamiltonian:
    :param paths:
    :param plot:
    :return:
    """
    LP = []
    LP_up = []
    LP_down = []
    for time_index in range(paths.shape[1]):
        mean = parametrized_ising_hamiltonian.log_probability(paths[:, time_index, :],True).item()
        std = parametrized_ising_hamiltonian.log_probability(paths[:, time_index, :],False).item()
        LP_up.append(mean + std)
        LP.append(mean)
        LP_down.append(mean - std)
    if plot:
        plt.plot(LP_up,"b-")
        plt.plot(LP)
        plt.plot(LP_down,"r-")
        plt.show()
    return LP

def spin_state_counts(x,find_all_states=False):
    """
    In order to counts the spins states, we enforce a lexicograpical order and then
    perform the cumulative sum over the consecutive equals

    :param x:
    :return:
    """
    lexicographical_x = nested_lexicographical_order(x)
    # zero means not equal
    equals = lexicographical_x[:-1, :] == lexicographical_x[1:, :]
    equals = torch.prod(equals, dim=1)

    if find_all_states:
        #all states
        new_states = torch.where(equals == 0)
        new_states = torch.cat([new_states[0], torch.tensor([new_states[0][-1] + 1])])
        all_states = x[new_states]

    # the states with more than one counts are given by
    # a one followed by a zero to the right (the last in a series of equals)
    equals_ = torch.cat([torch.Tensor([0]), equals, torch.Tensor([0])])
    cumsum = torch.cumsum(equals, dim=0)
    where_equals_1 = torch.where(equals_ == 1)[0]
    equals_1_shifted = equals_[where_equals_1 + 1]
    where_equals_1_shifted_0 = torch.where(equals_1_shifted == 0)[0]
    where_counted_states = where_equals_1[where_equals_1_shifted_0]
    counted_states = x[where_counted_states]
    counts = torch.cat([torch.Tensor([0]), cumsum[where_counted_states - 1]])
    counts = (counts[1:] - counts[:-1]) + 1.

    #unique states
    equals_ = torch.cat([equals, torch.Tensor([0])])
    if equals_.sum() == 0:
        unique_states = all_states
    else:
        where_equals_0 = torch.where(equals_[:-1] == 0)[0]
        equals_0_shifted = equals_[where_equals_0 + 1]
        where_equals_0_shifted_0 = torch.where(equals_0_shifted == 0)[0]
        where_unique_states = where_equals_0[where_equals_0_shifted_0]
        unique_states = x[where_unique_states + 1]

    if find_all_states:
        return all_states,unique_states, counted_states, counts
    else:
        return unique_states, counted_states, counts

def obtain_new_spin_states(counted_states,flip_mask):
    """
    flips one by one all the spins

    :param counted_states:
    :return:
    """
    number_of_counted_states,number_of_spins = counted_states.shape
    repeated_counted_states = torch.repeat_interleave(counted_states,number_of_spins,dim=0)
    repeated_mask = torch.tile(flip_mask,(number_of_counted_states,1))
    new_states = repeated_counted_states*repeated_mask
    return new_states

def obtain_new_states(counted_states,flip_mask):
    number_of_counted_states,number_of_spins = counted_states.shape
    repeated_counted_states = torch.repeat_interleave(counted_states,number_of_spins,dim=0)
    repeated_mask = torch.tile(flip_mask,(number_of_counted_states,1))
    new_states = repeated_counted_states*repeated_mask
    return new_states

def select_random_position(paths, time_grid):
    """
    Obtains the states and counts as well as the
    concatenation of the time embeddings
    """
    number_of_grid_points = time_grid.shape[0]
    random_time_in_grid = torch.randint(0, number_of_grid_points, (1,)).item()
    time_selected = time_grid[random_time_in_grid]
    x = paths[:, random_time_in_grid, :]
    unique_states, counted_states, counts = spin_state_counts(x)

    new_unique_states = obtain_new_states(unique_states)
    new_counted_states = obtain_new_states(counted_states)
    repeated_unique_states = torch.repeat_interleave(unique_states, number_of_spins, dim=0)
    repeated_counted_states = torch.repeat_interleave(counted_states, number_of_spins, dim=0)

    number_of_new_counted = new_counted_states.shape[0]
    number_of_new_unique = new_unique_states.shape[0]
    time_selected_repeated = torch.full((number_of_new_counted,), time_selected)
    time_counted_embedding = transformer_timestep_embedding(time_selected_repeated)

    time_selected_repeated = torch.full((number_of_new_unique,), time_selected)
    time_unique_embedding = transformer_timestep_embedding(time_selected_repeated)

    new_counted_n_time = torch.cat([new_counted_states, time_counted_embedding], dim=1)
    counted_n_time = torch.cat([repeated_counted_states, time_counted_embedding], dim=1)

    new_unique_n_time = torch.cat([new_unique_states, time_unique_embedding], dim=1)
    unique_n_time = torch.cat([repeated_unique_states, time_unique_embedding], dim=1)

    states = (repeated_unique_states,new_unique_states,repeated_counted_states,new_counted_states,counts)
    states_n_time = (counted_n_time, new_counted_n_time, unique_n_time, new_unique_n_time)

    return states, states_n_time

#=============================================================
# REVISED LEXICOGRAPHICAL ORDERING
#=============================================================

def obtain_all_spin_states(number_of_spins: int) -> torch.Tensor:
    """
    Obtains all possible states

    Parameters
    ----------
    number_of_spins: int

    Returns
    -------
    all_states
    """
    assert number_of_spins < 12, "More than 10 spins not accepted for obtaining all states"

    PATTERN = torch.Tensor([1., -1.]).unsqueeze(0).T

    def append_permutation(prefix, PATTERN=torch.Tensor([1., -1.]).unsqueeze(0).T):
        number_of_current_states = prefix.shape[0]
        PATTERN = PATTERN.repeat((number_of_current_states, 1))
        prefix = prefix.repeat_interleave(2, dim=0)
        return torch.hstack([prefix, PATTERN])

    for i in range(number_of_spins - 1):
        PATTERN = append_permutation(PATTERN)

    return PATTERN

def obtain_index_changes(x,depth_index):
    index_changes = torch.where(x[:-1, depth_index] != x[1:, depth_index])[0]
    index_changes = index_changes + 1
    index_change_0 = torch.tensor([0])
    index_changes = torch.cat([index_change_0,index_changes])
    index_changes = index_changes.tolist()
    index_changes.append(None)
    index_changes_ = []
    for i in range(len(index_changes)-1):
        index_changes_.append((index_changes[i],index_changes[i+1]))
    return index_changes_

def changes_and_box(x, prior_change, depth_index):
    index_changes = obtain_index_changes(x, depth_index)
    new_boxes = []
    for changes in index_changes:
        new_boxes.append((x[changes[0]:changes[1], :], prior_change + changes[0]))
    return new_boxes

def substates_to_change_from_index(x, latest_depth_index):
    new_boxes_0 = changes_and_box(x, 0, 0)
    for depth_index in range(latest_depth_index + 1):
        new_boxes_1 = []
        for box_ in new_boxes_0:
            if box_[0].shape[0] == 1:
                new_boxes_1.append(box_)
            else:
                new_boxes_ = changes_and_box(box_[0], box_[1], depth_index)
                new_boxes_1.extend(new_boxes_)
        new_boxes_0 = new_boxes_1
    return new_boxes_0

def order_at_index(x, depth_index_to_order=3):
    latest_depth_index = depth_index_to_order - 1
    boxes_to_change = substates_to_change_from_index(x, latest_depth_index)

    x_ordered = torch.clone(x)
    for box_and_index in boxes_to_change:
        box = box_and_index[0]
        index_ = box_and_index[1]
        box_lenght = box.shape[0]
        if box_lenght > 1:
            ordered_box = torch.clone(box)
            ordered_box_sort = torch.sort(ordered_box[:, depth_index_to_order])
            ordered_box = ordered_box[ordered_box_sort.indices, :]
            x_ordered[index_:index_ + box_lenght, :] = ordered_box
    return x_ordered

def nested_lexicographical_order(x):
    full_depth = x.shape[1]
    x_sort = torch.sort(x[:, 0])
    x = x[x_sort.indices, :]
    for latest_depth in range(1, full_depth):
        x = order_at_index(x, latest_depth)
    return x

def counts_states(x):
    # the states with more than one counts are given by
    # a one followed by a zero to the right (the last in a series of equals)

    lexicographical_x = nested_lexicographical_order(x)
    # zero means not equal
    equals = lexicographical_x[:-1, :] == lexicographical_x[1:, :]
    equals = torch.prod(equals, dim=1)
    equals_ = torch.cat([torch.Tensor([0]), equals])
    where_new = torch.where(equals_ == 0)
    different_states = lexicographical_x[where_new]

    # count all consecutive ones
    where_new = where_new[0].tolist()
    where_new.append(None)
    counts = []
    for i in range(len(where_new) - 1):
        (where_new[i], where_new[i + 1])
        counts.append(equals_[where_new[i]:where_new[i + 1]].sum().item())
    counts = torch.Tensor(counts)
    counts = counts + 1

    return different_states, counts

class spin_states_stats:
    """
    Class defined in order to obtain statistics for the paths,
    as well as obtaining the parametrized histogram version of the backward process

    this requieres the indexing of the states defined with the lexicographical ordering
    of all possible states, for 3 spins we have:

    0    [-1., -1., -1.],
    1    [-1., -1.,  1.],
    2    [-1.,  1., -1.],
    3    [-1.,  1.,  1.],
    4    [ 1., -1., -1.],
    5    [ 1., -1.,  1.],
    6    [ 1.,  1., -1.],
    7    [ 1.,  1.,  1.],

    so the markov transition probabilities are defined as Q_01, counting the number of transitions in the path
    that lead state 0 (as defined from the lexicographical ordering) to reach state 1.
    """
    def __init__(self,number_of_spins):
        self.number_of_spins = number_of_spins
        self.all_states_in_order = nested_lexicographical_order(obtain_all_spin_states(number_of_spins))
        self.number_of_total_states = self.all_states_in_order.shape[0]
        self.number_of_symetric_pairs = self.number_of_total_states*number_of_spins

        self.flip_mask = torch.ones((self.number_of_spins, self.number_of_spins))
        self.flip_mask.as_strided([self.number_of_spins], [self.number_of_spins + 1]).copy_(torch.ones(self.number_of_spins) * -1.)

        self.index_to_state = {}
        for i in range(self.number_of_total_states):
            self.index_to_state[i] = self.all_states_in_order[i]

        self.from_states_symmetric = torch.repeat_interleave(self.all_states_in_order, self.number_of_spins,dim=0)
        self.to_states_of_symmetric_function = obtain_new_spin_states(self.all_states_in_order, self.flip_mask)
        self.symmetric_function_pairs_()

    def state_index(self,state):
        return torch.where(torch.prod(self.all_states_in_order == state, dim=1))[0].item()

    def symmetric_function_pairs_(self):
        """
        The symmetric function as defined in the notes, only calculate transitions for one spin pair flips,
        here we identify does changes as pairs in the lexicographical ordering indexing
        """
        self.symmetric_function_pairs = []
        for i in range(self.number_of_symetric_pairs):
            i_from = self.state_index(self.from_states_symmetric[i])
            j_to = self.state_index(self.to_states_of_symmetric_function[i])
            self.symmetric_function_pairs.append((i_from, j_to))

    def symmetric_transition_part_(self,ising_schrodinger_trained):
        f_ss = ising_schrodinger_trained.symmetric_function_from_state(self.all_states_in_order)
        self.symmetric_transition_part = torch.zeros((self.number_of_total_states, self.number_of_total_states))
        for i in range(f_ss.shape[0]):
            i_from, j_to = self.symmetric_function_pairs[i]
            self.symmetric_transition_part[i_from, j_to] = f_ss[i]
        self.f_ss_with_indexing = f_ss

    def counts_for_different_states(self,states):
        different_states, counts = counts_states(states)
        number_of_different_states = different_states.shape[0]
        counts_for_different_states = torch.zeros((self.number_of_total_states,))
        all_states_index = 0
        for i in range(number_of_different_states):
            while not torch.equal(different_states[i], self.all_states_in_order[all_states_index]):
                all_states_index += 1
                if all_states_index > self.number_of_total_states:
                    break
            counts_for_different_states[all_states_index] = counts[i]
        return counts_for_different_states

    def counts_states_in_paths(self,paths):
        number_of_steps = paths.shape[1]
        counts_evolution = torch.zeros((number_of_steps, self.number_of_total_states))
        for time_index in range(number_of_steps):
            states = paths[:, time_index, :]
            counts_for_different_ = self.counts_for_different_states(states=states)
            counts_evolution[time_index, :] = counts_for_different_
        return counts_evolution

    def glauber_transition(self,ising_schrodinger):
        """
        returns
        states.shape[0]*states.shape[1]

        a vector where each states is changed one spin at a time
        """
        states = self.all_states_in_order
        states_repeated = states.repeat_interleave(self.number_of_spins, axis=0)
        i_selection = torch.tile(torch.arange(0, self.number_of_spins), (states.shape[0],))
        coupling_matrix = ising_schrodinger.obtain_couplings_as_matrix()
        J_i = coupling_matrix[:, i_selection].T
        H_i = ising_schrodinger.fields[i_selection]
        H_i = H_i + torch.einsum('bi,bi->b', J_i, states_repeated)
        x_i = torch.diag(states_repeated[:, i_selection])
        f_xy = (ising_schrodinger.mu * torch.exp(-x_i * H_i)) / 2 * torch.cosh(H_i)
        return f_xy

    def obtain_glauber_transition_matrix(self,ising_schrodinger):
        f_xy = self.glauber_transition(ising_schrodinger)
        self.glauber_matrix = torch.zeros((self.number_of_total_states, self.number_of_total_states))
        for i in range(f_xy.shape[0]):
            i_from, j_to = self.symmetric_function_pairs[i]
            self.glauber_matrix[i_from, j_to] = f_xy[i]
        self.glauber_with_indexing = f_xy

    def obtain_markov_transition_matrices(self,paths):
        number_of_steps = paths.shape[1]
        markov_step_matrices = torch.zeros(
            (number_of_steps-1, self.number_of_total_states, self.number_of_total_states))
        for time_index in range(number_of_steps - 1):
            step = paths[:, time_index:time_index + 2, :]
            for state_index in range(self.number_of_total_states):
                current_state = self.all_states_in_order[state_index]
                where_state = torch.where(torch.prod((step[:, 0, :] == current_state), dim=1))
                states_reached = step[where_state[0], 1, :]
                count_states_reached = self.counts_for_different_states(states_reached)
                markov_step_matrices[time_index, state_index, :] = count_states_reached/count_states_reached.sum()
        return markov_step_matrices

    def obtain_backward_transition_matrices(self,paths):
        forward_transition_matrices = self.obtain_markov_transition_matrices(paths)
        number_of_steps = forward_transition_matrices.shape[0]
        backward_transition_matrices = torch.zeros_like(forward_transition_matrices)
        for tau in range(1, number_of_steps + 1):
            backward_transition_matrices[number_of_steps - tau] = inv(forward_transition_matrices[number_of_steps - tau])
        return backward_transition_matrices


#=============================================================
# ARGUMENTS
#=============================================================

number_of_spins = 4

kwargs = {
    "number_of_spins": 4,
    "J_mean": 0,
    "J_std": 1.,
    "mu": 0.9,
    "T": 3.,
    "tau": 0.09,
    "p0": [0.2] * number_of_spins,
    "p1": [0.8] * number_of_spins,
    "time_embedding_dim": 10,
    "number_of_paths": 20,
    "max_number_of_time_steps": 20000,
    "number_of_estimate_iterations": 3000,
    "epsilon_threshold": 1e-3,
    "number_of_sinkhorn_iterations": 30
}

number_of_spins = kwargs.get("number_of_spins")
J_mean = kwargs.get("J_mean")
J_std = kwargs.get("J_std")
mu = kwargs.get("mu")

phi_parameters = {
    "input_dim": number_of_spins,
    "output_dim": 1,
    "layers_dim": [50, 50],
    "output_transformation": None,
    "dropout": 0.4
}

T = kwargs.get("T")
tau = kwargs.get("tau")
time_embedding_dim = kwargs.get("time_embedding_dim")

p0 = torch.Tensor(kwargs.get("p0"))
p1 = torch.Tensor(kwargs.get("p1"))

number_of_paths = kwargs.get("number_of_paths")
max_number_of_time_steps = kwargs.get("max_number_of_time_steps")
number_of_estimate_iterations = kwargs.get("number_of_estimate_iterations")
number_of_sinkhorn_iterations = kwargs.get("number_of_sinkhorn_iterations")
epsilon_threshold = kwargs.get("epsilon_threshold")

phi_parameters['input_dim'] = number_of_spins + time_embedding_dim

rows_index = torch.arange(0, number_of_paths)
time_grid = torch.arange(0., T, tau)

flip_mask = torch.ones((number_of_spins, number_of_spins))
flip_mask.as_strided([number_of_spins], [number_of_spins + 1]).copy_(torch.ones(number_of_spins) * -1.)

