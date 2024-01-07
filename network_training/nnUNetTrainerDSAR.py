#########################################################################################################
# -------------------This class represents the nnUNet trainer for DSAR training.----------------------#
#########################################################################################################


import torch
import torch.nn.functional as F
import torchvision.transforms as T
from time import time
from nnunet_ext.paths import default_plans_identifier
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from batchgenerators.utilities.file_and_folder_operations import *
from training.network_training.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead
import numpy as np
from torch.cuda.amp import autocast
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from loss_functions.deep_supervision import MultipleOutputLossDSARCDA as CDALoss
from network_architecture.modules import VectorQuantizedVAE

# -- Define globally the Hyperparameters for this trainer along with their type -- #

HYPERPARAMS = {'dsar_lambda': float, 'cda_beta': float}

def randconv(image: torch.Tensor, K: int, mix: bool, p: float) -> torch.Tensor:
    """
    Outputs the image or the random convolution applied on the image.

    Args:
        image (torch.Tensor): input image
        K (int): maximum kernel size of the random convolution
    """

    p0 = torch.rand(1).item()
    if p0 < p:
        return image
    else:
        k = torch.randint(1, K+1, (1, )).item()
        random_convolution = nn.Conv2d(3, 3, 2*k + 1, padding=k).to(image.device)
        torch.nn.init.uniform_(random_convolution.weight,
                            0, 1. / (3 * k * k))
        image_rc = random_convolution(image).to(image.device)

        if mix:
            alpha = torch.rand(1,)
            alpha = alpha.to(image.device)
            return alpha * image + (1 - alpha) * image_rc
        else:
            return image_rc
import random

class nnUNetTrainerDSAR(nnUNetTrainerMultiHead):
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='ewc', dsar_lambda=10, cda_beta= 1e-2,  tasks_list_with_char=None, 
                 mixed_precision=True, save_csv=True, del_log=False, use_vit=True, vit_type='base', version=1, split_gpu=False, 
                 transfer_heads=False, ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, network=None, use_param_split=False):
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice,  stage, unpack_data, 
                         deterministic, fp16, save_interval, already_trained_on, use_progress, identifier, extension, 
                         dsar_lambda, cda_beta, tasks_list_with_char, mixed_precision, save_csv, del_log, use_vit, vit_type,
                         version, split_gpu, transfer_heads, ViT_task_specific_ln, do_LSA, do_SPT, network, use_param_split)

        self.dsar_data_path = join(self.trained_on_path, 'dsar_cda_codebook')
        maybe_mkdir_p(self.dsar_data_path)
        self.gmm = dict()
        
        if 'eye' in self.dataset_directory:
            self.num_classes = 3
            self.num_channels = 3
            self.label_ids = {
                'background': 0,
                'disc': 1,
                'cup':2
            }
        else:
            self.num_classes = 2
            self.num_channels = 2
            self.label_ids = {
                'background': 0,
                'prostate': 1,
            }

        # -- Initialize vqvae model -- #
        self.task = self.dataset_directory.split('/')[-1]
        self.shepe_codebook_channel = 1
        self.shepe_codebook_hidden_size = 128
        self.shepe_codebook_K = 512
        self.shepe_codebook_device = 'cuda:0'
        self.shape_codebook_visualize = False
        self.shape_codebook_visualize_save_path = './vqvae/save/prostate_visualize/'
        self.model_save_path = './vqvae/save/models/CDBv2_h128k512/'
        if 'eye' in self.dataset_directory:
            self.model_save_path = './vqvae/save/models/CDBv2_h128k512_eye/'


    def initialize(self, tasks_list_with_char, training=True, force_load_plans=False, num_epochs=500, prev_trainer_path=None,
                   call_for_eval=False):
        r"""Overwrite the initialize function so the correct Loss function for the EWC method can be set.
        """
        # -- Perform initialization of parent class -- #
        super().initialize(training, force_load_plans, num_epochs, prev_trainer_path, call_for_eval)

        # -- Reset self.loss from MultipleOutputLoss2 to DC_and_CE_loss so the EWC Loss can be initialized properly -- #
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

        # -- Choose the right loss function (EWC) that will be used during training -- #
        # -- --> Look into the Loss function to see how the approach is implemented -- #
        # -- Update the network paramaters during each iteration -- #
        self.loss = CDALoss(self.loss, self.ds_loss_weights,
                            self.dsar_lambda,
                            self.cda_beta,
                            self.fisher,
                            self.params,
                            self.gmm,
                            self.network.named_parameters(),
                            num_channels = self.num_channels,
                            num_classes = self.num_classes)

        # -- Load trained vqvae model -- # 
        if not training: 
            self.task = self.already_trained_on['0']['finished_training_on'][-1]
        self.vqvae = VectorQuantizedVAE(self.shepe_codebook_channel, self.shepe_codebook_hidden_size, self.shepe_codebook_K).to(self.shepe_codebook_device)
        model_path = self.model_save_path + self.task + '/best.pt'
        self.print_to_log_file('==> Load vqvae model weights:' + model_path)
        model_state = torch.load(model_path, map_location=self.shepe_codebook_device)
        self.vqvae.load_state_dict(model_state)
        for _, param in self.vqvae.named_parameters():
            param.requires_grad = False
        self.vqvae.eval()

    def reinitialize(self, task):
        # -- Execute the super function -- #
        super().reinitialize(task)
        # -- Print Loss update -- #
        self.print_to_log_file("I am using DSAR-CDA loss now")
        
        data_dict = next(self.tr_gen)
        target = data_dict['target']
        n_samples = np.zeros(self.num_classes, dtype=int)
        cls, ns = np.unique(target[0].detach().cpu().numpy(), return_counts=True)
        for i in range(len(cls)):
            n_samples[int(cls[i])] = ns[i]

        self.loss.update_gmm_params(self.gmm, n_samples)

        # -- Upload vqvae model -- #
        self.task = self.dataset_directory.split('/')[-1]
        model_path = self.model_save_path + self.task + '/best.pt'
        self.print_to_log_file('==> Load vqvae model weights:' + model_path)
        model_state = torch.load(model_path, map_location=self.shepe_codebook_device)
        self.vqvae.load_state_dict(model_state)
        for _, param in self.vqvae.named_parameters():
            param.requires_grad = False
        self.vqvae.eval()

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, detach=True, no_loss=False):
        r"""This function runs an iteration based on the underlying model. It returns the detached or undetached loss.
            The undetached loss might be important for methods that have to extract gradients without always copying
            the run_iteration function.
            NOTE: The calling class needs to set self.network according to the desired task, this is not done in this
                  function but expected by the user.
        """
        # -- Run iteration as usual --> copied and modified from nnUNetTrainerV2 -- #
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        mask = target[0]
        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = 0
                zs = 0
                if (self.num_channels == 32):
                    output, reg_output , zs = self.network.forward_features_32(data, self.vqvae)
                elif (self.num_channels == 64):
                    output, reg_output , zs = self.network.forward_features_64(data, self.vqvae)
                else:
                    output, reg_output , zs = self.network.forward_features(data, self.vqvae)
                del data
                if not no_loss:
                    reference_seg_output, reg_gt, _ = self.vqvae(mask)
                    l = self.loss(output, target, reg_gt, reg_output, zs)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        # -- Update the Multi Head Network after one iteration only if backprop is performed (during training) -- #
        if do_backprop:
            self.mh_network.update_after_iteration()

        loss = ''

        # -- Return the loss -- #
        if not no_loss:
            if detach:
                loss = l.detach().cpu().numpy()
                # return loss

        # -- After running one iteration and calculating the loss, update the parameters of the loss for the next iteration -- #
        # -- NOTE: The gradients DO exist even after the loss detaching of the super function, however the loss function -- #
        # --       does not need them, since they are only necessary for the Fisher values that are calculated once the -- #
        # --       training is done performing an epoch with no optimizer steps --> see after_train() for that -- #
        self.loss.update_network_params(self.network.named_parameters())

        # -- Return the loss -- #
        if not no_loss:
            return loss

        
    def after_train(self):
        r"""This function needs to be executed once the training of the current task is finished.
            The function will use the same data to generate the gradients again and setting the
            models parameters.
        """
        # gen_samples for training is no use for this site; clear it to save GPU memory!
        self.loss.gen_samples.clear()
        self.loss.gen_samples_eval.clear()

        # -- Update the log -- #
        self.print_to_log_file("Running one last epoch without changing the weights to extract Fisher and Parameter values...")
        start_time = time()
        #------------------------------------------ Partially copied from original implementation ------------------------------------------#
        # -- Put the network in train mode and kill gradients -- #
        self.network.train()
        self.optimizer.zero_grad()

        for _ in range(self.num_batches_per_epoch):
            T1, T2, T3 = 0, 0, 0
            T = 8
            while (T1+T2+T3 != 8):
                T1 = random.randint(1, T-2)
                T2 = random.randint(1, T-2)
                T3 = random.randint(1, T-2)
                
            self.optimizer.zero_grad()
            # -- Extract the data -- #
            data_dict = next(self.tr_gen)
            data = data_dict['data']
            target = data_dict['target']

            # -- Push data to GPU -- #
            data = maybe_to_torch(data)
            target = maybe_to_torch(target)
            if torch.cuda.is_available():
                data = to_cuda(data)
                target = to_cuda(target)
            
            domain_specific_loss = 0
            # -- Respect the fact if the user wants to autocast during training -- #
            if self.fp16:
                with autocast():
                    output = self.network(data)
                    # brightness, contrast, saturation
                    jitter = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
                    aug_datas = [jitter(data) for _ in range(T1)]
                    for data_pass in aug_datas:
                        seg_pass = self.network(data_pass)
                        domain_specific_loss += F.mse_loss(output[0], seg_pass[0])
                        del seg_pass
                    # texture
                    for _ in range(T2):
                        data_pass = randconv(data, K=9, mix=True, p=0)
                        seg_pass = self.network(data_pass)
                        domain_specific_loss += F.mse_loss(output[0], seg_pass[0])
                        del data_pass, seg_pass
                    # gauss noise
                    for _ in range(T3):
                        data_pass = data
                        data_pass.copy_(data)
                        data_pass += (0.1**0.5)*torch.randn_like(data_pass)
                        seg_pass = self.network(data_pass)
                        domain_specific_loss += F.mse_loss(output[0], seg_pass[0])
                        del data_pass, seg_pass
                        
                self.network.zero_grad()
                domain_specific_loss.backward()
                # -- Set fisher and params in current fold from last iteration --> final model parameters -- #
                for name, param in self.network.named_parameters():
                    # -- Update the fisher dict -- #
                    if param.grad is not None:
                        if name in self.fisher[self.task].keys():
                            self.fisher[self.task][name] += (param.grad.data.clone().abs()/self.num_batches_per_epoch)
                        else:
                            self.fisher[self.task][name] = (param.grad.data.clone().abs()/self.num_batches_per_epoch)

            del data, target, domain_specific_loss

        # -- Set fisher and params in current fold from last iteration --> final model parameters -- #
        for name, param in self.network.named_parameters():
            if param.grad is None:
                self.fisher[self.task][name] = torch.tensor([1], device='cuda:0')
            # -- Update the params dict -- #
            self.params[self.task][name] = param.data.clone()
            
        # -- Update the log -- #
        self.print_to_log_file("Extraction and saving of Fisher and Parameter values took %.2f seconds" % (time() - start_time))

        task = self.task
        start_time = time()
        means, _, ct =  self.learn_gaussians()
        self.print_to_log_file("computed means in ", time() - start_time)
        start_time = time()
        means, covs, ct = self.learn_gaussians(initial_means=means)
        self.print_to_log_file("finished training gaussians in", time() - start_time)
        np.save(self.dsar_data_path + '/' + task + "_means.npy", means)
        np.save(self.dsar_data_path + '/' + task + "_covs.npy", covs)

        self.gmm[task] = dict()
        self.gmm[task]['means'] = means
        self.gmm[task]['covs'] = covs
        self.print_to_log_file(self.gmm)

    def learn_gaussians(self, rho=0.95, initial_means=None):
        self.print_to_log_file(
            "Running one last epoch without changing the weights to extract GMM means and convs...")
        label_ids = self.label_ids
        num_classes = len(label_ids)
        num_channels = self.num_channels
        # -- find the network body and head -- #
        self.network.eval()
        means = initial_means
        if initial_means is None:
            means = np.zeros((num_classes, num_channels))
        covs = np.zeros((num_classes, num_channels, num_channels))
        cnt = np.zeros(num_classes)
        
        # -- Do loop through the data based on the number of batches -- 
        for _ in range(self.num_batches_per_epoch):
            self.optimizer.zero_grad()
            # -- Extract the data -- #
            data_dict = next(self.tr_gen)
            data = data_dict['data']
            target = data_dict['target']

            # -- Push data to GPU -- #
            data = maybe_to_torch(data)
            target = maybe_to_torch(target)
            if torch.cuda.is_available():
                data = to_cuda(data)
                target = to_cuda(target)
            # model predict
            y_hat = 0
            # embedding features
            zs = 0
            # features before softmax
            zss = 0
            # Predict latent features
            if (num_channels == 32):
                y_hat, _, zs = self.network.forward_features_32(data, self.vqvae)
                _, _, zs = self.network.forward_features(data, self.vqvae)
            elif (num_channels == 64):
                y_hat, _, zs = self.network.forward_features_64(data, self.vqvae)
                _, _, zs = self.network.forward_features(data, self.vqvae)
            else:
                y_hat, _, zs = self.network.forward_features(data, self.vqvae)
                zss = zs

            y_softmax = F.softmax(zss, dim=1)
            zs = zs.transpose(1,3).reshape(-1, num_channels)

            y_hat = y_hat[0].transpose(1,3).reshape(-1,num_classes)
            y_hat = y_hat.cpu().detach().numpy()
            y_hat = np.argmax(y_hat, axis=-1)

            y_softmax = y_softmax.transpose(1,3).reshape(-1,num_classes)
            y_softmax = y_softmax.cpu().detach().numpy()
            y_softmax  = np.max(y_softmax, axis=-1)

            y_t = target[0].transpose(1,3).reshape(-1,1)
            y_t = y_t.cpu().detach().numpy()
            y_t = y_t.ravel()
            del data,target

            # Keep a few exemplars per class 
            for label in label_ids:
                # if label == 'ignore':
                #     continue 
                c = label_ids[label]  # class id
                ind = (y_t == c) & (y_hat == c) & (y_softmax > rho)
                if np.sum(ind) > 0:
                    # We have at least one sample
                    # Reshape to make sure dimensions stay the same
                    curr_data = zs[ind].reshape(-1, num_channels)
                    curr_data = curr_data.cpu().detach().numpy()
                    print(curr_data.shape)
                    if initial_means is None:
                        # Only update means and counts
                        means[c] += np.sum(curr_data, axis=0)
                        cnt[c] += np.sum(ind)
                    else:
                        # ! here means are scaled to their final values
                        sigma = np.dot(np.transpose(curr_data - means[c]), curr_data - means[c])
                        assert sigma.shape == (num_channels, num_channels)
                        covs[c] += sigma
                        cnt[c] += np.sum(ind)

            del zs, y_hat, y_t

        # Normalize results
        for i in range(num_classes):
            if initial_means is None:
                means[i] /= cnt[i]
            covs[i] /= (cnt[i] - 1)
        return means, covs, cnt