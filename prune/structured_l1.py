import numpy as np

import torch
import torch.nn as nn

import models


def ifcontain(name, keywords):
    print(name)
    for keyword in keywords:
        if keyword in name.split("."):
            print(name)
            return True
    return False

class structured_L1:


    def __init__(self, model, pruning_rate=0.25, skip = []):
        """ Init pruning method """
        self.pruning_rate = pruning_rate
        self.model = model

        self.type = "global"
        # init masks
        self.cfg = []
        self.cfg_mask = []
        self.masks = []
        self.skip = skip


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
        self.prune_both = []
        convList = []
        print("show names")

        for name, m in self.model.named_modules():
            print(name)

        for m in self.model.modules():

            if isinstance(m, nn.Conv2d):
                convList.append(m)

        print("step!!!!!")
        print(convList)


        self.cfg = []
        self.cfg_mask = []

        for index in range(len(convList)):
            m = convList[index]
            in_channels = m.weight.data.shape[1]
            out_channels = m.weight.data.shape[0]

            if index in self.skip:
                self.cfg.append(out_channels)
                continue


            if index == 0 and  out_channels == convList[index+1].weight.data.shape[1]:
                if index+1 not in self.skip:
                    self.prune_out.append(index)
                    weight_copy = m.weight.data.abs().clone().cpu().numpy()
                    L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
                    num_keep = int(out_channels * (1 - self.pruning_rate))
                    arg_max = np.argsort(L1_norm)
                    arg_max_rev = arg_max[::-1][:num_keep]
                    mask = torch.zeros(out_channels)
                    mask[arg_max_rev.tolist()] = 1
                    self.cfg_mask.append(mask)
                    self.cfg.append(num_keep)
                else:
                    print("next conv need to skip,this convID: {} next convID : {} name :{}".format(index, index+1, convList[index+1][0]))
                    self.cfg.append(out_channels)
                continue

            if index+1 == len(convList) and in_channels == convList[index-1].weight.data.shape[0]:
                self.prune_in.append(index)
                self.cfg.append(out_channels)
                continue

            if out_channels == convList[index+1].weight.data.shape[1]:

                weight_copy = m.weight.data.abs().clone().cpu().numpy()
                L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
                num_keep = int(out_channels * (1 - self.pruning_rate))
                arg_max = np.argsort(L1_norm)
                arg_max_rev = arg_max[::-1][:num_keep]
                mask = torch.zeros(out_channels)
                mask[arg_max_rev.tolist()] = 1
                self.cfg_mask.append(mask)
                self.cfg.append(num_keep)
                if index-1 in self.skip or in_channels != convList[index-1].weight.data.shape[0]:
                    self.prune_out.append(index)
                else:
                    self.prune_both.append(index)

                continue


            if out_channels != convList[index+1].weight.data.shape[1] and in_channels == convList[index-1].weight.data.shape[0]:

                self.prune_in.append(index)
                self.cfg.append(out_channels)
                continue

            self.cfg.append(out_channels)
        print("prune_out : {}".format(self.prune_out))
        print("prune_in : {}".format(self.prune_in))
        print("prune_both : {}".format(self.prune_both))
        print(len(self.cfg_mask))

    def zero_params(self,masks=None):
        """ Apply the masks, ie, zero out the params """
        masks = masks if masks is not None else self.cfg_mask

        print('zeroing')
        layer_id_in_cfg = 0
        # mask_out = torch.ones(3)
        # mask_int = torch.ones(3)

        conv_count = 0
        bn_count = 0
        module_list = list(self.model.modules())
        for index in range(len(module_list)):
            m0 = module_list[index]

            if isinstance(m0, nn.Conv2d):
                # print("conv2d #{}, in_channel = {}, out_channel = {}".format(conv_count, m0.in_channels, m0.out_channels))
                # print("weight shape:")
                # print(m0.weight.shape)
                if conv_count in self.prune_out:
                    print("prune_out conv2d ID : {}".format(conv_count))
                    out_channel = self.cfg[conv_count]
                    mask_out = masks[layer_id_in_cfg]
                    idx_out = np.squeeze(np.argwhere(np.asarray(mask_out.cpu().numpy())))
                    if idx_out.size == 1:
                        idx_out = np.resize(idx_out, (1,))

                    w = m0.weight.data[idx_out.tolist(), :, :, :].clone()

                    m0.out_channels = out_channel
                    m0.weight.data = w.clone()

                    if isinstance(module_list[index+1], nn.BatchNorm2d):
                        m0 = module_list[index+1]
                        m0.num_features = out_channel
                        m0.weight.data = m0.weight.data[idx_out.tolist()].clone()

                        m0.bias.data = m0.bias.data[idx_out.tolist()].clone()
                        m0.running_mean = m0.running_mean[idx_out.tolist()].clone()
                        m0.running_var = m0.running_var[idx_out.tolist()].clone()
                        index += 1

                    layer_id_in_cfg += 1
                    print("layer_id :{}".format(layer_id_in_cfg))
                    conv_count += 1
                    continue

                if conv_count in self.prune_both:
                    print("prune_both conv2d ID : {}".format(conv_count))
                    out_channel = self.cfg[conv_count]
                    in_channel = self.cfg[conv_count-1]
                    mask_out = masks[layer_id_in_cfg]
                    mask_in = masks[layer_id_in_cfg - 1]
                    idx_out = np.squeeze(np.argwhere(np.asarray(mask_out.cpu().numpy())))
                    if idx_out.size == 1:
                        idx_out = np.resize(idx_out, (1,))
                    idx_in = np.squeeze(np.argwhere(np.asarray(mask_in.cpu().numpy())))
                    if idx_in.size == 1:
                        idx_in = np.resize(idx_in, (1,))

                    w = m0.weight.data[idx_out.tolist(),:, :, :].clone()
                    w = w[:,idx_in.tolist() ,:,:].clone()
                    m0.out_channels = out_channel
                    m0.in_channels = in_channel

                    m0.weight.data = w.clone()
                    if isinstance(module_list[index+1], nn.BatchNorm2d):
                        m0 = module_list[index + 1]
                        m0.num_features = out_channel
                        m0.weight.data = m0.weight.data[idx_out.tolist()].clone()

                        m0.bias.data = m0.bias.data[idx_out.tolist()].clone()
                        m0.running_mean = m0.running_mean[idx_out.tolist()].clone()
                        m0.running_var = m0.running_var[idx_out.tolist()].clone()
                    layer_id_in_cfg += 1
                    print("layer_id :{}".format(layer_id_in_cfg))
                    conv_count += 1

                    continue

                if conv_count in self.prune_in:
                    print("prune_in conv2d ID : {}".format(conv_count))
                    in_channel = self.cfg[conv_count - 1]
                    mask = masks[layer_id_in_cfg - 1]
                    idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                    if idx.size == 1:
                        idx = np.resize(idx, (1,))
                    w = m0.weight.data[:, idx.tolist(), :, :].clone()

                    m0.in_channels = in_channel
                    m0.weight.data = w.clone()
                    conv_count += 1
                    continue


                conv_count += 1

