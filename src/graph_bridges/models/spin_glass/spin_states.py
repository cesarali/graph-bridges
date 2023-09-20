import torch
from torch.linalg import inv

#=============================================================
# REVISED LEXICOGRAPHICAL ORDERING
#=============================================================
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

class SpinStatesStatistics:
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

    def state_index(self,state):
        return torch.where(torch.prod(self.all_states_in_order == state, dim=1))[0].item()

    def spins_state_legend(self):
        legend = []
        for spin_state_index in range(self.number_of_total_states):
            legend.append(str(self.all_states_in_order[spin_state_index].int().tolist()))
        return legend

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
        """
        :param paths:
                torch.Tensor(batch_size,number_of_timesteps,number_of_spins)
        :return: counts_evolution
                torch.Tensor(number_of_timesteps,number_of_total_states)
        """
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

if __name__=="__main__":
    from discrete_diffusion.utils.postprocessing.experiments_files_utils import obtain_models_from_results
    from discrete_diffusion.utils.postprocessing.experiments_files_utils import get_results_dictionary
    from discrete_diffusion.data.paths_dataloaders import PathsDataloader

    # OBTAIN RESULTS OF EXPERIMENTS ------------------------------------------------------------
    RESULTS = get_results_dictionary(1680775110)
    backward_model, forward_model, paths_dataloader = obtain_models_from_results(RESULTS)
    path_batch, times = next(paths_dataloader.train().__iter__())

    spin_statistics = SpinStatesStatistics(paths_dataloader.spin_dataloader_0.number_of_spins)
    histogram_of_states = torch.zeros(paths_dataloader.number_of_time_steps+1,spin_statistics.number_of_total_states)
    for path_batch, times in paths_dataloader.train():
        counts_per_states = spin_statistics.counts_states_in_paths(path_batch)
        histogram_of_states+=counts_per_states

