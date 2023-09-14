import torch

#========================================================================
# ONE STEP ESTIMATOR
#========================================================================

class BackwardOneStepEstimator:
    name_ = "backward_one_shot_estimator"

    def __init__(self,**kwargs):
        self.parameters_ = kwargs
        self.dimension = kwargs.get("dimension")
        self.parameters_.update({"name": self.name_})
        self.device = torch.device("cpu")

    def set_device(self,device):
        self.device = device

    def obtain_parameters(self):
        return self.parameters_

    def estimator(self,current_model,past_model,paths_batch,current_time):
        """
        :param current_model:
        :param X_spins:
        :return:
        """
        assert paths_batch.shape[1] == 2
        X_spins_0 = paths_batch[:,0,:]
        X_spins_t = paths_batch[:,1,:]

        X_spins_0_hat = current_model.one_back_regression(X_spins_t,current_time)
        loss = (X_spins_0_hat - X_spins_0)**2.

        if self.dimension is not None:
            loss_d = loss[:, self.dimension]
            return loss_d.mean()
        else:
            return loss.mean()