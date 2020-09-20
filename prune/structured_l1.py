import numpy as np

import torch
import torch.nn as nn

import models

class structured_L1:
    """ L1 based pruning with an optimizer-like interface  """


    def __init__(self, model, pruning_rate=0.25, skip = [], depth = None):
        """ Init pruning method """
        self.pruning_rate = pruning_rate
        self.model = model
        self.skip = skip
        self.type = "global"
        # init masks
        self.cfg = []
        self.cfg_mask = []
        self.masks = []
        self.depth = depth

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

    def get_cfg(self):
        return self.cfg

    def get_type(self):
        return self.type

    ##############
    # Core methods
    def step(self):
        """ Update the pruning masks """
        self.prune_out = []
        self.prune_in = []
        out = True



        layer_id = 0
        self.cfg = []
        self.cfg_mask = []
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                out_channels = m.weight.data.shape[0]
                if layer_id in self.skip:
                    layer_id += 1
                    continue
                if out:
                    if self.depth:
                        gate = self.depth / 3
                        if layer_id <=gate:
                            prune_prob_stage = self.pruning_rate[0]
                        elif layer_id <= gate *2:
                            prune_prob_stage = self.pruning_rate[1]
                        else:
                            prune_prob_stage = self.pruning_rate[2]
                    else:
                        prune_prob_stage = self.pruning_rate
                    print("pruning ratio: {}".format(prune_prob_stage))
                    weight_copy = m.weight.data.abs().clone().cpu().numpy()
                    L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
                    num_keep = int(out_channels * (1 - prune_prob_stage))
                    arg_max = np.argsort(L1_norm)
                    arg_max_rev = arg_max[::-1][:num_keep]
                    mask = torch.zeros(out_channels)
                    mask[arg_max_rev.tolist()] = 1
                    self.cfg_mask.append(mask)
                    self.cfg.append(num_keep)

                    out = False
                    self.prune_out.append(layer_id)
                    layer_id += 1
                    continue
                else:
                    self.prune_in.append(layer_id)
                    out = True
                    layer_id += 1
                    continue
        print("prune_out : {}".format(self.prune_out))
        print("prune_in : {}".format(self.prune_in))
        print(len(self.cfg_mask))

    def zero_params(self, new_model,masks=None):
        """ Apply the masks, ie, zero out the params """
        masks = masks if masks is not None else self.cfg_mask

        print('zeroing')
        layer_id_in_cfg = 0
        mask = torch.ones(3)
        conv_count = 0
        bn_count = 0

        for m0, m1 in zip(self.model.modules(), new_model.modules()):
            if isinstance(m0, nn.Conv2d):
                # print("conv2d #{}, in_channel = {}, out_channel = {}".format(conv_count, m0.in_channels, m0.out_channels))
                # print("weight shape:")
                # print(m0.weight.shape)
                if conv_count in self.prune_out:
                    print("conv2d ID : {}".format(conv_count))
                    mask = masks[layer_id_in_cfg]
                    idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                    if idx.size == 1:
                        idx = np.resize(idx, (1,))

                    w = m0.weight.data[idx.tolist(), :, :, :].clone()
                    m1.weight.data = w.clone()
                    layer_id_in_cfg += 1
                    print("layer_id :{}".format(layer_id_in_cfg))
                    conv_count += 1

                    continue
                if conv_count in self.prune_in:
                    print("conv2d ID : {}".format(conv_count))
                    mask = masks[layer_id_in_cfg - 1]
                    idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                    if idx.size == 1:
                        idx = np.resize(idx, (1,))
                    w = m0.weight.data[:, idx.tolist(), :, :].clone()
                    m1.weight.data = w.clone()
                    conv_count += 1
                    continue
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
            elif isinstance(m0, nn.BatchNorm2d):
                if bn_count in self.prune_out:
                    #print("in_channel: {}, out_channel:{}".format(m0.weight.data.shape[1],m0.weight.data.shape[0]))
                    mask = masks[layer_id_in_cfg - 1]
                    idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                    if idx.size == 1:
                        idx = np.resize(idx, (1,))
                    m1.weight.data = m0.weight.data[idx.tolist()].clone()
                    m1.bias.data = m0.bias.data[idx.tolist()].clone()
                    m1.running_mean = m0.running_mean[idx.tolist()].clone()
                    m1.running_var = m0.running_var[idx.tolist()].clone()
                    bn_count += 1
                    continue
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
                bn_count += 1


            elif isinstance(m0, nn.Linear):
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
        return new_model

    ##############
