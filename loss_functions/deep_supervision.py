#########################################################################################################
# ---------------------Corresponding deep_supervision.py file for DSAR method.---------------------#
#########################################################################################################

import torch, copy
import torch.nn as nn
import torch.nn.functional as F
from nnunet.utilities.to_torch import to_cuda
from nnunet_ext.training.loss_functions.embeddings import *
from nnunet_ext.training.loss_functions.crossentropy import *
from nnunet_ext.training.loss_functions.knowledge_distillation import *
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
import numpy as np
from scipy.stats import wasserstein_distance
import random


class MultipleOutputLossDSAR(MultipleOutputLoss2):
    def __init__(self, loss, weight_factors=None, dsar_lambda=10,
                 fisher=dict(), params=dict(), network_params=None, match_sth=False, match=list(), match_true=True):
        super(MultipleOutputLossDSAR, self).__init__(loss, weight_factors)
        # -- Set all variables that are used by parent class and are necessary for the EWC loss calculation -- #
        self.weight_factors = weight_factors
        self.loss = loss
        self.dsar_lambda = dsar_lambda
        self.tasks = list(fisher.keys())
        self.fisher = fisher
        self.params = params
        self.network_params = network_params
        self.match_case = match_sth
        self.match = match
        self.match_true = match_true

    def update_ewc_params(self, fisher, params):
        r"""The ewc parameters should be updated after every finished run before training on a new task.
        """
        # -- Update the parameters -- #
        self.tasks = list(fisher.keys())
        self.fisher = fisher
        self.params = params

    def update_network_params(self, network_params):
        r"""The network parameters should be updated after every finished iteration of an epoch, 
            in order for the loss to use always the current network parameters. --> use this in
            run_iteration function before the loss will be calculated.
        """
        # -- Update the network_params -- #
        self.network_params = network_params

    def forward(self, x, y, reg_gt, reg_output, reg=True):
        # -- Segmentation Loss -- #
        loss = super(MultipleOutputLossDSAR, self).forward(x, y)
        # -- Shape Codebook Loss -- #
        cdb_loss = F.mse_loss(reg_gt, reg_output)
        loss += cdb_loss
        # -- DSAR regularzation Loss -- #
        if reg:  # Do regularization ?
            for name, param in self.network_params:  # Get named parameters of the current model
                for task in self.tasks:
                    # -- Only consider those parameters in which the matching phrase is matched if desired or all -- #
                    param_ = to_cuda(param, gpu_id=loss.get_device())
                    fisher_value = to_cuda(self.fisher[task][name], gpu_id=loss.get_device())
                    param_value = to_cuda(self.params[task][name], gpu_id=loss.get_device())
                    loss += self.dsar_lambda / 2 * (fisher_value * (param_ - param_value).pow(2)).sum()

        # -- Return the updated loss value -- #
        return loss


class MultipleOutputLossDSARCDA(MultipleOutputLossDSAR):
    def __init__(self, loss, weight_factors=None, dsar_lambda=10, cda_beta=1e-2, fisher=dict(), params=dict(),
                 gmm_means_covs=dict(), network_params=None, num_channels=2, num_classes=2,  match_sth=False, match=list(), match_true=True):
        super(MultipleOutputLossDSARCDA, self).__init__(loss, weight_factors, dsar_lambda, fisher, params, network_params,
                                                    match_sth, match, match_true)
        self.cda_beta = cda_beta
        self.gmm_means_covs = gmm_means_covs
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.gen_samples = []
        self.gen_samples_eval = []
        self.pseudo_sample_pool_size = 128
        if self.num_channels == 32:
            self.pseudo_sample_pool_size = 8
        if self.num_channels == 64:
            self.pseudo_sample_pool_size = 4
        self.train_batch_size = 8
        self.eval_batch_size = 4

    def update_gmm_params(self, gmm_means_covs, n_samples):
        r"""The ewc parameters should be updated on the fly following EWC++ method.
        """
        # -- Update the parameters -- #
        self.gmm_means_covs = gmm_means_covs
        self.n_samples = n_samples
        self.gen_samples = []
        self.gen_samples_eval = []
        
        # -- generate pseudo samples -- #
        feature_shape = (384, 384, self.num_channels)

        if len(self.gmm_means_covs.keys()) > 0:
            key = random.randint(0, len(self.gmm_means_covs.keys()))
            site = list(self.gmm_means_covs.keys())[key-1]
            means = self.gmm_means_covs[site]['means']
            covs = self.gmm_means_covs[site]['covs']
            for _ in range(self.pseudo_sample_pool_size):
                # for training
                Yembed_list = []
                Yembed_list_eval = []
                for _ in range(self.train_batch_size):
                    Yembed, _ = self.sample_from_gaussians(means, covs, n_samples=n_samples)
                    Yembed = Yembed.reshape(-1, feature_shape[0], feature_shape[1], self.num_channels)
                    Yembed = np.transpose(Yembed, [0,3,1,2])
                    Yembed_list.append(Yembed)
                Yembed = np.vstack(Yembed_list)
                Yembed = torch.from_numpy(Yembed)
                Yembed = to_cuda(Yembed)
                self.gen_samples.append(Yembed)
                # for evaluation
                for _ in range(self.eval_batch_size):
                    Yembed, _ = self.sample_from_gaussians(means, covs, n_samples=n_samples)
                    Yembed = Yembed.reshape(-1, feature_shape[0], feature_shape[1], self.num_channels)
                    Yembed = np.transpose(Yembed, [0,3,1,2])
                    Yembed_list.append(Yembed)
                Yembed_eval = np.vstack(Yembed_list_eval)
                Yembed_eval = torch.from_numpy(Yembed_eval)
                Yembed_eval = to_cuda(Yembed_eval)
                self.gen_samples_eval.append(Yembed_eval)

    def forward(self, x, y, reg_gt, reg_output, zs, reg=True):
        # -- Calculate the loss first using the parent class -- #
        loss = super(MultipleOutputLossDSARCDA, self).forward(x, y, reg_gt, reg_output)
        matching_loss = 0
        if len(self.gmm_means_covs.keys())>0:
            zs = zs.detach()
            pseudo_sample_idx = random.randint(0, self.pseudo_sample_pool_size-1)
            yembed = 0
            if(zs.shape[0] == self.train_batch_size):
                yembed = self.gen_samples[pseudo_sample_idx]
            else:
                yembed = self.gen_samples_eval[pseudo_sample_idx]
            yembed = yembed.half()
            zs = zs.half()
            distance = self.cda_beta * self.discrepancy_slice_wasserstein(yembed, zs)
            matching_loss += distance
        return loss + matching_loss

    def sample_from_gaussians(self, means, covs, n_samples):
        # Return samples from the num_channels gaussians trained on the source dataset
        # n_samples is an array of the same size as gmms, representing the number of samples to
        # be returned from each gaussian
        # class 0 is the class to be ignored
        n = len(n_samples)
        res_x = []
        res_y = []
        for i in range(n):
            curr_x = np.random.multivariate_normal(means[i], covs[i], n_samples[i])
            curr_y = np.repeat(i, n_samples[i])
            res_x.append(curr_x)
            res_y.append(curr_y.reshape(-1, 1))
        res_x = np.vstack(res_x)
        res_y = np.vstack(res_y).ravel()
        perm = np.random.permutation(res_x.shape[0])
        return res_x[perm, :], res_y[perm]
    
    def discrepancy_slice_wasserstein(self, p1, p2):
        channel = p1.shape[1]
        p1 = p1.transpose(1,3).reshape(-1,channel)
        p2 = p2.transpose(1,3).reshape(-1,channel)
        s = p1.shape
        proj = torch.randn(channel, 64) # 64 random projection dimensions
        proj = to_cuda(proj)
        proj *= torch.rsqrt(torch.sum(torch.mul(proj, proj), 0, keepdim=True))
        p1 = torch.matmul(p1, proj) 
        p2 = torch.matmul(p2, proj)
        p1 = torch.topk(p1, s[0], dim=0)[0]
        p2 = torch.topk(p2, s[0], dim=0)[0]
        dist = p1-p2
        wdist = torch.mean(torch.mul(dist, dist))
        return wdist