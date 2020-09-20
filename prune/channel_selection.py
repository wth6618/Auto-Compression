import numpy as np

import torch
import torch.nn as nn

import models


class Channel_selection:
    """ CS pruning with an optimizer-like interface  """

    def __init__(self, model, pruning_rate=0.25):
        """ Init pruning method """
        self.pruning_rate = float(pruning_rate)
        self.model = model

        # init masks
        self.masks = []
        self.cfg = []

    ################################################
    # Reporting nonzero entries and number of params
    def count_nonzero(self):
        """ Count nonzero elements """
        return sum(mask.sum() for mask in self.masks)

    def numel(self):
        """ Number of elements """
        return int(sum(mask.view(-1).size(0) for mask in self.masks))
    ################################################

    ############################################
    # Methods for resetting or rewinding params
    def clone_params(self):
        """ Copy all tracked params, such that they we can rewind to them later """
        return [p.clone() for p in self.model.parameters()]

    def rewind(self, cloned_params):
        """ Rewind to previously stored params """
        for p_old, p_new in zip(self.model.parameters(), cloned_params):
            p_old.data = p_new.data
    ############################################

    ##############
    # Core methods
    def step(self):
        """ Update the pruning masks """

        total = 0

        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                total += m.weight.data.shape[0]

        bn = torch.zeros(total)
        index = 0
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                bn[index:(index + size)] = m.weight.data.abs().clone()
                index += size

        y, i = torch.sort(bn)
        thre_index = int(total * self.pruning_rate)
        thre = y[thre_index]

        pruned = 0
        cfg = []
        self.masks = []
        for k, m in enumerate(self.model.modules()):
            if isinstance(m, nn.BatchNorm2d):
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(thre).float().cuda()
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                cfg.append(int(torch.sum(mask)) if int(torch.sum(mask)) != 0 else 1)
                self.masks.append(mask.clone())
                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                      format(k, mask.shape[0], int(torch.sum(mask))))
            elif isinstance(m, nn.MaxPool2d):
                cfg.append('M')

        print("Pruned Ratio: ", pruned/total)

    def zero_params(self, masks=None):
        """ Apply the masks, ie, zero out the params """
        masks = masks if masks is not None else self.masks

        modules = list(self.model.modules())
        layer_id_in_cfg = 0
        start_mask = torch.ones(3)
        end_mask = masks[layer_id_in_cfg]
        conv_count = 0

        for layer_id in range(len(modules)):
            m0 = modules[layer_id]
            if isinstance(m0, nn.BatchNorm2d):
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))

                if isinstance(modules[layer_id + 1], models.channel_selection):
                    # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.
                    # We need to set the channel selection layer.
                    m2 = modules[layer_id + 1]
                    m2.indexes.data.zero_()
                    m2.indexes.data[idx1.tolist()] = 1.0

                    layer_id_in_cfg += 1
                    start_mask = end_mask.clone()
                    if layer_id_in_cfg < len(masks):
                        end_mask = masks[layer_id_in_cfg]
                else:
                    m0.weight.data = m0.weight.data[idx1.tolist()]
                    m0.bias.data = m0.bias.data[idx1.tolist()]
                    m0.running_mean = m0.running_mean[idx1.tolist()]
                    m0.running_var = m0.running_var[idx1.tolist()]
                    layer_id_in_cfg += 1
                    start_mask = end_mask.clone()
                    if layer_id_in_cfg < len(masks):  # do not change in Final FC
                        end_mask = masks[layer_id_in_cfg]
            elif isinstance(m0, nn.Conv2d):
                if conv_count == 0:
                    m0.weight.data = m0.weight.data.clone()
                    conv_count += 1
                    continue
                if isinstance(modules[layer_id - 1], models.channel_selection) or isinstance(
                        modules[layer_id - 1], nn.BatchNorm2d):
                    # This convers the convolutions in the residual block.
                    # The convolutions are either after the channel selection layer or after the batch normalization layer.
                    conv_count += 1
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()

                    # If the current convolution is not the last convolution in the residual block, then we can change the
                    # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
                    if conv_count % 3 != 1:
                        w1 = w1[idx1.tolist(), :, :, :].clone()
                    m0.weight.data = w1.clone()
                    continue

                # We need to consider the case where there are downsampling convolutions.
                # For these convolutions, we just copy the weights.

            elif isinstance(m0, nn.Linear):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))

                m0.weight.data = m0.weight.data[:, idx0]

    ##############
