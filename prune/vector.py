import torch
import torch.nn as nn
import torch.nn.functional as F


def cut(pruning_rate, flat_params):
    """ Compute cutoff value within `flat_params` at percentage `pruning_rate`."""
    assert flat_params.dim() == 1
    # Compute cutoff value
    with torch.no_grad():
        cutoff_index = round(pruning_rate * flat_params.size()[0])
        values, __indices = torch.sort(torch.abs(flat_params))
        cutoff = values[cutoff_index]
    return cutoff


class VectorPruning():
    """ Magnitude pruning with an optimizer-like interface  """

    def __init__(self, model, pruning_rate=0.80, layer_wise=False, global_wise=False, target_layers=None):
        """ Init pruning method """
        self.layer_wise = layer_wise
        self.global_wise = global_wise
        self.target_layers = target_layers
        self.pruning_rate = float(pruning_rate)

        # if exclude_biases:
        #     # Discover all non-bias parameters
        #     self.params = [p for p in params if p.dim() > 1]
        # else:
        #     self.params = [p for p in params]
        self.model = model

        self.masks = []
        # # init masks to all ones
        # masks = []
        # for p in self.params:
        #     masks.append(torch.ones_like(p))
        #
        # self.masks = masks

    ################################################
    # Reporting nonzero entries and number of params
    def count_nonzero(self):
        """ Count nonzero elements """
        return int(sum(mask.sum() for mask in self.masks).item())

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
        if self.global_wise:
            total, num_rows = 0, 0

            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    total += m.weight.data.numel()
                    num_rows +=  m.out_channels * m.kernel_size[0] * m.in_channels

            conv_weights = torch.zeros(num_rows)
            index = 0
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    row = m.out_channels * m.kernel_size[0] * m.in_channels
                    row_sum = m.weight.data.abs().clone().sum((3), keepdim = True)
                    conv_weights[index:(index + row)] = row_sum.flatten()
                    index += row

            y, i = torch.sort(conv_weights)
            thre_index = int(num_rows * self.pruning_rate)
            thre = y[thre_index]
            pruned = 0
            print('Pruning threshold: {}'.format(thre))
            for k, m in enumerate(self.model.modules()):
                if isinstance(m, nn.Conv2d):
                    weight_copy = m.weight.data.abs().clone()
                    mask = weight_copy.sum((3), keepdim = True).gt(thre).float().cuda()
                    pruned = pruned + mask.numel() - torch.sum(mask)
                    self.masks.append(mask)

                    if int(torch.sum(mask)) == 0:
                        zero_flag = True
                    print('layer index: {:d} \t total rows: {:d} \t remaining rows: {:d}'.
                          format(k, mask.numel(), int(torch.sum(mask))))
            print(
                'Total conv params rows: {}, Pruned conv params rows: {}, Pruned ratio: {}'.format(num_rows, pruned, pruned / num_rows))

        if self.layer_wise:

            assert self.target_layers, "Please enter the Layers you want to prune"
            print("layer wise pruning")
            total = 0
            pruned = 0
            print(self.target_layers)

            index = 0
            layer_id, pruning_ratio = self.target_layers[index]
            for k, m in enumerate(self.model.modules()):
                if k == layer_id:
                    if isinstance(m, nn.Conv2d):
                        #size = m.weight.data.numel()
                        size = m.out_channels * m.kernel_size[0] * m.in_channels
                        #conv_weights = torch.zeros(size)
                        row_sum = m.weight.data.abs().clone().sum((3), keepdim = True)
                        y, _ = torch.sort(row_sum.flatten())
                        thre_index = int(size * pruning_ratio)
                        thre = y[thre_index]

                        print('Pruning threshold: {}'.format(thre))
                        weight_copy = m.weight.data.abs().clone()
                        mask = weight_copy.sum((3), keepdim = True).gt(thre).float().cuda()
                        pruned = pruned + mask.numel() - torch.sum(mask)
                        self.masks.append(mask)

                        if int(torch.sum(mask)) == 0:
                            zero_flag = True
                        print('layer index: {:d} \t total params rows: {:d} \t remaining params rows: {:d}'.
                              format(k, mask.numel(), int(torch.sum(mask))))
                    else:
                        print("layer {} is not Conv2d \n {}".format(k, m))
                    index += 1
                    if index >= len(self.target_layers):
                        break
                    layer_id, pruning_ratio = self.target_layers[index]

            # print(
            #     'Total conv params in layer {} : {}, Pruned conv params: {}, Pruned ratio: {}'.format(layer_id,size, pruned, pruned / size))
            # for i, (m, p) in enumerate(zip(self.masks, self.model.modules())):
            #     # Compute cutoff
            #     if isinstance()
            #     flat_params = p[m == 1].view(-1)
            #     cutoff = cut(self.pruning_rate, flat_params)
            #     # Update mask
            #     new_mask = torch.where(torch.abs(p) < cutoff,
            #                            torch.zeros_like(p), m)
            #     self.masks[i] = new_mask
        # else:  # Global pruning #
        #
        #     # Gather all masked parameters
        #     flat_params = torch.cat([p[m == 1].view(-1)
        #                              for m, p in zip(self.masks, self.params)])
        #     # Compute cutoff value
        #     cutoff = cut(self.pruning_rate, flat_params)
        #
        #     # Calculate updated masks
        #     for i, (m, p) in enumerate(zip(self.masks, self.params)):
        #         new_mask = torch.where(torch.abs(p) < cutoff,
        #                                torch.zeros_like(p), m)
        #         self.masks[i] = new_mask

    def zero_params(self, masks=None):
        """ Apply the masks, ie, zero out the params """
        masks = masks if masks is not None else self.masks
        # for m, p in zip(masks, self.params):
        #     p.data = m * p.data
        layer = 0
        if self.layer_wise:
            layer_id, _ = self.target_layers.pop(0)
            for k, m in enumerate(self.model.modules()):
                if isinstance(m, nn.Conv2d) and k == layer_id:
                    # print('layer {}, mask: {}'.format(m, masks[layer]))
                    m.weight.data.mul_(masks[layer])
                    layer += 1

                if self.target_layers:
                    layer_id, pruning_ratio = self.target_layers.pop(0)
                else:
                    break
        if self.global_wise:
            for k, m in enumerate(self.model.modules()):
                if isinstance(m, nn.Conv2d):
                    # print('layer {}, mask: {}'.format(m, masks[layer]))
                    m.weight.data.mul_(masks[layer])
                    layer += 1

    ##############
# Global
# TEST RESULTS
# {'Accuracy': 93.86}
# --------------------------------------------------------------------------------
# Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:09<00:00,  4.44it/s]
# Pruning threshold: 0.0012141085462644696
# layer index: 3   total rows: 576         remaining rows: 576
# layer index: 9   total rows: 4096        remaining rows: 3598
# layer index: 11          total rows: 12288       remaining rows: 12265
# layer index: 13          total rows: 16384       remaining rows: 12860
# layer index: 17          total rows: 16384       remaining rows: 13237
# layer index: 20          total rows: 16384       remaining rows: 12371
# layer index: 22          total rows: 12288       remaining rows: 12198
# layer index: 24          total rows: 16384       remaining rows: 10331
# layer index: 28          total rows: 16384       remaining rows: 12880
# layer index: 30          total rows: 12288       remaining rows: 12188
# layer index: 32          total rows: 16384       remaining rows: 10639
# layer index: 37          total rows: 32768       remaining rows: 26667
# layer index: 39          total rows: 49152       remaining rows: 48795
# layer index: 41          total rows: 65536       remaining rows: 48603
# layer index: 45          total rows: 131072      remaining rows: 91050
# layer index: 48          total rows: 65536       remaining rows: 42425
# layer index: 50          total rows: 49152       remaining rows: 48782
# layer index: 52          total rows: 65536       remaining rows: 46820
# layer index: 56          total rows: 65536       remaining rows: 48432
# layer index: 58          total rows: 49152       remaining rows: 48940
# layer index: 60          total rows: 65536       remaining rows: 44493
# layer index: 64          total rows: 65536       remaining rows: 49834
# layer index: 66          total rows: 49152       remaining rows: 48810
# layer index: 68          total rows: 65536       remaining rows: 36811
# layer index: 73          total rows: 131072      remaining rows: 95091
# layer index: 75          total rows: 196608      remaining rows: 189267
# layer index: 77          total rows: 262144      remaining rows: 127329
# layer index: 81          total rows: 524288      remaining rows: 149720
# layer index: 84          total rows: 262144      remaining rows: 85206
# layer index: 86          total rows: 196608      remaining rows: 171527
# layer index: 88          total rows: 262144      remaining rows: 63152
# layer index: 92          total rows: 262144      remaining rows: 82893
# layer index: 94          total rows: 196608      remaining rows: 156858
# layer index: 96          total rows: 262144      remaining rows: 36483
# layer index: 100         total rows: 262144      remaining rows: 57918
# layer index: 102         total rows: 196608      remaining rows: 93581
# layer index: 104         total rows: 262144      remaining rows: 20069
# layer index: 108         total rows: 262144      remaining rows: 57856
# layer index: 110         total rows: 196608      remaining rows: 86566
# layer index: 112         total rows: 262144      remaining rows: 24160
# layer index: 116         total rows: 262144      remaining rows: 57484
# layer index: 118         total rows: 196608      remaining rows: 101588
# layer index: 120         total rows: 262144      remaining rows: 43009
# layer index: 125         total rows: 524288      remaining rows: 39559
# layer index: 127         total rows: 786432      remaining rows: 175930
# layer index: 129         total rows: 1048576     remaining rows: 99056
# layer index: 133         total rows: 2097152     remaining rows: 87328
# layer index: 136         total rows: 1048576     remaining rows: 6890
# layer index: 138         total rows: 786432      remaining rows: 94053
# layer index: 140         total rows: 1048576     remaining rows: 77117
# layer index: 144         total rows: 1048576     remaining rows: 3810
# layer index: 146         total rows: 786432      remaining rows: 86779
# layer index: 148         total rows: 1048576     remaining rows: 66365
# Total conv params rows: 15901248, Pruned conv params rows: 12720999.0, Pruned ratio: 0.8000000715255737
# Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:07<00:00, 12.02it/s]-
# -------------------------------------------------------------------------------
# TEST RESULTS
# {'Accuracy': 93.23}


# Local
# layers = [(120, 0.9), (125, 0.9), (138, 0.9), (140, 0.95)]
# TEST RESULTS
# {'Accuracy': 93.86}
# --------------------------------------------------------------------------------
# Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:09<00:00,  4.39it/s]
# layer wise pruning
# [(120, 0.9), (125, 0.9), (138, 0.9), (140, 0.95)]
# Pruning threshold: 0.0017784038791432977
# layer index: 120         total params rows: 262144       remaining params rows: 26214
# Pruning threshold: 0.0010467255488038063
# layer index: 125         total params rows: 524288       remaining params rows: 52428
# Pruning threshold: 0.0013513454468920827
# layer index: 138         total params rows: 786432       remaining params rows: 78643
# Pruning threshold: 0.0017091594636440277
# layer index: 140         total params rows: 1048576      remaining params rows: 52428
# Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:07<00:00, 11.99it/s]-
# -------------------------------------------------------------------------------
# TEST RESULTS
# {'Accuracy': 93.86}
# --------------------------------------------------------------------------------
# Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:07<00:00,  5.25it/s]
#
