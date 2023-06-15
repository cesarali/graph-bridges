import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def plot_of_training_loss(LOSS, windows_size=10):
    if type(LOSS) in [list, np.ndarray]:
        # Create a time series
        values = pd.Series(LOSS)
        # Compute the rolling average
        rolling_average = values.rolling(windows_size).mean()
        # Plot the time series with the rolling average
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(values, alpha=0.3)
        ax.plot(rolling_average, color="red")
        ax.set_xlabel("Iter")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        plt.show()
    else:
        raise Exception("LOSS not array")

if __name__=="__main__":
    import os
    from discrete_diffusion import results_path as results_dir

    all_experiments_dir = os.path.join(results_dir, 'ratio_estimator')
    experiment_dir = os.path.join(all_experiments_dir, "dimension_{0}".format(1680775110))
    best_results_path = os.path.join(experiment_dir, "best_model.tr")
    RESULTS = torch.load(best_results_path)

    plot_of_training_loss(RESULTS["LOSS"])