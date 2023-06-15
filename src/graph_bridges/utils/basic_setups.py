import os
import time
import shutil
from discrete_diffusion import results_path
from torch.utils.tensorboard import SummaryWriter

def create_dir_and_writer(model_name="particles_schrodinger",
                          experiments_class="ou",
                          model_identifier=None,
                          delete=False,
                          sinkhorn=False):
    if model_identifier is None:
        model_identifier = str(int(time.time()))
    my_results_dir = os.path.join(results_path, model_name, "{0}_{1}".format(experiments_class,model_identifier))
    if not os.path.isdir(my_results_dir):
        os.makedirs(my_results_dir)
    else:
        if delete:
            shutil.rmtree(my_results_dir)
            os.makedirs(my_results_dir)

    tensorboard_path = os.path.join(my_results_dir, "tensorboard")

    if os.path.isdir(tensorboard_path):
        shutil.rmtree(tensorboard_path)
    writer = SummaryWriter(tensorboard_path)

    if sinkhorn:
        best_model_path = os.path.join(my_results_dir, "sinkhorn_{0}.tr")
    else:
        best_model_path = os.path.join(my_results_dir, "best_model.tr")

    if sinkhorn:
        sinkhorn_plot_path = os.path.join(my_results_dir, "marginal_at_site_sinkhorn_{0}.png")
        return writer,my_results_dir,best_model_path,model_identifier, sinkhorn_plot_path
    else:
        return writer, my_results_dir, best_model_path, model_identifier