from matplotlib import pyplot as plt

def sinkhorn_plot(sinkhorn_iteration,
                  states_histogram_at_0,
                  states_histogram_at_1,
                  backward_histogram,
                  forward_histogram,
                  time_,
                  states_legends,
                  max_number_of_states_displayed=8,
                  save_path=None):
    """
    Forward is the direction of the past model

    :param is_past_forward:
    :param time_:
    :param states_histogram_at_0:
    :param states_histogram_at_1:
    :param histogram_from_rate:
    :param states_legends:
    :return:
    """
    start_target = backward_histogram[0,:]
    end_target = forward_histogram[-1,:]
    number_of_total_states = states_histogram_at_0.shape[0]
    # create the layout with GridSpec

    fig, axs = plt.subplots(figsize=(12, 3),
                            nrows=1, ncols=3,
                            gridspec_kw={'width_ratios': [1, 5, 1]},)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.35, top=0.8, hspace=0.1)

    ax1 = axs[0]
    ax1.set_title(r"$P_0(x)$")
    ax2 = axs[1]
    ax2.set_title("Sinkhorn Iteration {0} (-) Backward (*) Forward".format(sinkhorn_iteration))
    ax3 = axs[2]
    ax3.set_title(r"$P_T(x)$")

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for spin_state_index in range(min(max_number_of_states_displayed,number_of_total_states)):
        state_spins_vector = states_legends[spin_state_index]
        state_label = "{0}:".format(spin_state_index) + " " + str(state_spins_vector)
        color = colors[spin_state_index % len(colors)]
        # create the main plot
        ax2.plot(time_, backward_histogram[:, spin_state_index], "-", label=state_label, alpha=0.4, color=color)
        ax2.plot(time_, forward_histogram[:, spin_state_index], "*", alpha=0.4, color=color)

    ax2.legend(loc='upper center',bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=4)


    ax1.bar(range(number_of_total_states), states_histogram_at_0.tolist(),
            alpha=0.3,label="Data 0 ",color=colors[0])
    ax1.bar(range(number_of_total_states), start_target.tolist(),
            alpha=0.3,label="Backward",color=colors[1])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=1)

    ax3.bar(range(number_of_total_states), states_histogram_at_1.tolist(),alpha=0.3,
            label="Target T",color=colors[0])
    ax3.bar(range(number_of_total_states), end_target.tolist(), alpha=0.3,
            label="Forward",color=colors[1])
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=1)

    ax1.set_ylim(0.,1.)
    ax3.set_ylim(0.,1.)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)