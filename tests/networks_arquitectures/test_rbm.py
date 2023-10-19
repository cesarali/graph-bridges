import os
import torch
import unittest

class TestRBM(unittest.TestCase):
    """

    """

    @unittest.skip
    def test_rbm(self):
        from graph_bridges.models.networks_arquitectures.rbf import RBMConfig
        from graph_bridges.models.networks_arquitectures.rbf import RBM

        batch_size = 2
        num_hidden = 10
        num_visible = 20

        rbf_config = RBMConfig(num_visible=num_visible,
                               num_hidden=num_hidden)

        rbf = RBM(rbf_config)
        data_ = torch.rand((batch_size,num_visible))
        h1,ph1 = rbf(data_)
        v1,pv1 = rbf.backward(h1)

        print(h1.shape)
        print(v1.shape)

if __name__ == '__main__':
    unittest.main()