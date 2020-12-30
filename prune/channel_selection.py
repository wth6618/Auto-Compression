import numpy as np

import torch
import torch.nn as nn

import models
import collections


def getNewMask(m, num_keep):
    size = m.weight.data.shape[0]
    bn = m.weight.data.abs().clone()

    y, i = torch.sort(bn)

    thre = y[size - num_keep - 1]
    mask = bn.gt(thre).float().cuda()
    print(int(torch.sum(mask)))
    return mask


def ifcontain(name, keywords):
    for keyword in keywords:
        if keyword in name.split("."):
            return True
    return False


def getBlockID(name):
    splited = name.split(".")
    id = splited[0] if len(splited) < 2 else splited[0] + splited[1]
    return id


class Network_Slimming:

    def __init__(self, model, pruning_rate=0.25, keyword=["shortcut", "downsample"], bottleneck=4):
        """ Init pruning method """
        self.pruning_rate = pruning_rate
        self.model = model

        # init masks
        self.cfg = []
        self.cfg_mask = []
        self.keyword = keyword
        self.bottleneck = bottleneck

    ################################################
    # Reporting nonzero entries and number of params
    def count_nonzero(self):
        """ Count nonzero elements """
        return sum(mask.sum() for mask in self.cfg_masks)

    def numel(self):
        """ Number of elements """
        return int(sum(mask.view(-1).size(0) for mask in self.cfg_masks))

    ################################################

    ############################################
    # Methods for resetting or rewinding params
    def clone_params(self):
        """ Copy all tracked params, such that they we can rewind to them later """
        return [p.clone() for p in self.model.parameters()]

    def getTotal(self):
        return self.total

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
        self.skip = []
        self.basicBlock = collections.defaultdict(int)
        key = self.keyword
        convList = []
        bnList = []

        bn_total = 0
        convList = []
        total_channel = 0
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                convList.append((name, m))
                self.basicBlock[getBlockID(name)] = self.basicBlock[getBlockID(name)] + 1
                total_channel += m.out_channels
            if isinstance(m, nn.BatchNorm2d):
                bnList.append(m)
                # bn_total += m.weight.data.shape[0]

        print(len(bnList), len(convList))
        assert len(bnList) == len(convList), "unequal bn and conv"

        for index in range(len(convList)):

            name, layer = convList[index]

            outchannel = layer.out_channels
            inchannel = layer.in_channels
            # print("index:{}, name: {}, layer: {}, inchannel :{}, outchannel: {} ".format(index, name, layer, outchannel, inchannel))

            if ifcontain(name, key) or index == 0:
                self.skip.append(index)
                continue

            if index + 1 >= len(convList):
                if inchannel == convList[index - 1][1].out_channels:
                    self.prune_in.append(index)
                    continue
                else:
                    self.skip.append(index)
                    continue
            if outchannel != convList[index + 1][1].in_channels and inchannel == convList[index - 1][1].out_channels:
                self.prune_in.append(index)
                continue

            if outchannel == convList[index + 1][1].in_channels and inchannel != convList[index - 1][1].out_channels:
                self.prune_out.append(index)
                continue

            if outchannel == convList[index + 1][1].in_channels and inchannel == convList[index - 1][1].out_channels:
                if index - 1 in self.prune_in or index - 1 in self.skip:
                    # if ifcontain(convList[index+1][0], key) or getBlockID(name) != getBlockID(convList[index+1][0]):
                    if ifcontain(convList[index + 1][0], key):
                        self.skip.append(index)
                        continue
                    self.prune_out.append(index)
                    continue

                if self.basicBlock[getBlockID(name)] > 1 and getBlockID(name) != getBlockID(convList[index + 1][0]):
                    # if getBlockID(name) != getBlockID(convList[index+1][0]):
                    self.prune_in.append(index)
                    continue

                self.prune_both.append(index)
                continue
            self.skip.append(index)

        # print("prune_skip : {}".format(self.skip))
        # print("prune_out : {}".format(self.prune_out))
        # print("prune_in : {}".format(self.prune_in))
        # print("prune_both : {}".format(self.prune_both))

        for i in range(len(bnList)):
            if i not in self.skip and i not in self.prune_in:
                bn_total += bnList[i].weight.data.shape[0]

        bn = torch.zeros(bn_total)
        index = 0
        for i in range(len(bnList)):
            if i not in self.skip and i not in self.prune_in:
                m = bnList[i]
                size = m.weight.data.shape[0]
                bn[index:(index + size)] = m.weight.data.abs().clone()
                index += size
        # for m in bnList:
        #     size = m.weight.data.shape[0]
        #     bn[index:(index+size)] = m.weight.data.abs().clone()
        #     index += size

        y, i = torch.sort(bn)

        thre_index = int(bn_total * self.pruning_rate)
        thre = y[thre_index]

        self.total = bn_total
        self.cfg = []
        self.cfg_mask = []
        for index in range(len(convList)):
            _, m = convList[index]
            in_channels = m.weight.data.shape[1]
            out_channels = m.weight.data.shape[0]

            if index in self.skip:
                self.cfg.append((in_channels, out_channels))
                continue

            if index in self.prune_in:
                new_inchannel = self.cfg[-1][1]
                self.cfg.append((new_inchannel, out_channels))

                self.cfg_mask.append(self.cfg_mask[-1])
                continue

            if index in self.prune_out or index in self.prune_both:

                m_bn = bnList[index]
                weight_copy = m_bn.weight.data.abs().clone()
                mask = weight_copy.gt(thre).float().cuda()

                if int(torch.sum(mask)) < self.bottleneck:
                    mask = getNewMask(m_bn, self.bottleneck)
                m_bn.weight.data.mul_(mask)
                m_bn.bias.data.mul_(mask)

                self.cfg_mask.append(mask)
                if index in self.prune_both:
                    self.cfg.append((self.cfg[-1][1], int(torch.sum(mask))))
                else:
                    self.cfg.append((in_channels, int(torch.sum(mask))))

    def zero_params(self, masks=None):
        """ Apply the masks, ie, zero out the params """
        masks = masks if masks is not None else self.cfg_mask

        layer_id_in_cfg = 0

        conv_count = 0
        bn_count = 0
        module_list = list(self.model.modules())

        for index in range(len(module_list)):
            m0 = module_list[index]

            if isinstance(m0, nn.Conv2d):
                if conv_count in self.prune_out:

                    _, out_channel = self.cfg[conv_count]
                    mask_out = masks[layer_id_in_cfg]
                    idx_out = np.squeeze(np.argwhere(np.asarray(mask_out.cpu().numpy())))
                    if idx_out.size == 1:
                        idx_out = np.resize(idx_out, (1,))

                    w = m0.weight.data[idx_out.tolist(), :, :, :].clone()

                    m0.out_channels = out_channel

                    m0.weight.data = w.clone()
                    if m0.bias != None:
                        m0.bias.data = m0.bias.data[idx_out.tolist()].clone()

                    # if isinstance(module_list[index+1], nn.BatchNorm2d):
                    #     m0 = module_list[index+1]
                    #     m0.num_features = out_channel
                    #     m0.weight.data = m0.weight.data[idx_out.tolist()].clone()

                    #     m0.bias.data = m0.bias.data[idx_out.tolist()].clone()
                    #     m0.running_mean = m0.running_mean[idx_out.tolist()].clone()
                    #     m0.running_var = m0.running_var[idx_out.tolist()].clone()
                    #     index += 1

                    layer_id_in_cfg += 1

                    conv_count += 1
                    continue

                if conv_count in self.prune_both:

                    in_channel, out_channel = self.cfg[conv_count]

                    mask_out = masks[layer_id_in_cfg]
                    mask_in = masks[layer_id_in_cfg - 1]
                    idx_out = np.squeeze(np.argwhere(np.asarray(mask_out.cpu().numpy())))
                    if idx_out.size == 1:
                        idx_out = np.resize(idx_out, (1,))
                    idx_in = np.squeeze(np.argwhere(np.asarray(mask_in.cpu().numpy())))
                    if idx_in.size == 1:
                        idx_in = np.resize(idx_in, (1,))

                    w = m0.weight.data[idx_out.tolist(), :, :, :].clone()
                    w = w[:, idx_in.tolist(), :, :].clone()
                    m0.out_channels = out_channel
                    m0.in_channels = in_channel

                    m0.weight.data = w.clone()
                    if m0.bias != None:
                        m0.bias.data = m0.bias.data[idx_out.tolist()].clone()
                    # if isinstance(module_list[index+1], nn.BatchNorm2d):
                    #     m0 = module_list[index + 1]
                    #     m0.num_features = out_channel
                    #     m0.weight.data = m0.weight.data[idx_out.tolist()].clone()

                    #     m0.bias.data = m0.bias.data[idx_out.tolist()].clone()
                    #     m0.running_mean = m0.running_mean[idx_out.tolist()].clone()
                    #     m0.running_var = m0.running_var[idx_out.tolist()].clone()
                    layer_id_in_cfg += 1
                    # print("layer_id :{}".format(layer_id_in_cfg))
                    conv_count += 1

                    continue

                if conv_count in self.prune_in:
                    # print("prune_in conv2d ID : {}".format(conv_count))
                    in_channel, _ = self.cfg[conv_count]
                    mask = masks[layer_id_in_cfg - 1]
                    idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                    if idx.size == 1:
                        idx = np.resize(idx, (1,))
                    w = m0.weight.data[:, idx.tolist(), :, :].clone()

                    m0.in_channels = in_channel
                    m0.weight.data = w.clone()
                    conv_count += 1
                    layer_id_in_cfg += 1
                    continue

                conv_count += 1
            if isinstance(m0, nn.BatchNorm2d):
                if (conv_count - 1 in self.prune_out or conv_count - 1 in self.prune_both):
                    _, out_channel = self.cfg[conv_count - 1]
                    mask_out = masks[layer_id_in_cfg - 1]
                    idx_out = np.squeeze(np.argwhere(np.asarray(mask_out.cpu().numpy())))
                    m0.num_features = out_channel
                    m0.weight.data = m0.weight.data[idx_out.tolist()].clone()

                    m0.bias.data = m0.bias.data[idx_out.tolist()].clone()
                    m0.running_mean = m0.running_mean[idx_out.tolist()].clone()
                    m0.running_var = m0.running_var[idx_out.tolist()].clone()

