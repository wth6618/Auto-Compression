import torch
import torch.nn as nn
import torch.nn.functional as F


class Unstructured():
    """ Magnitude pruning with an optimizer-like interface  """

    def __init__(self, model, pruning_rate=0.25, layer_wise=False, global_wise=True, target_layers=None):
        """ Init pruning method """
        self.layer_wise = layer_wise
        self.global_wise = global_wise
        self.target_layers = target_layers
        self.pruning_rate = float(pruning_rate)
        self.model = model
        self.masks = []


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
            total, nonzero = 0 , 0
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    total += m.weight.data.numel()
                    temp = m.weight.data.abs().clone().gt(0).float().cuda()
                    nonzero += torch.sum(temp)
            conv_weights = torch.zeros(total)
            index = 0
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    size = m.weight.data.numel()
                    conv_weights[index:(index + size)] = m.weight.data.view(-1).abs().clone()
                    index += size

            y, i = torch.sort(conv_weights)
            thre_index = int(total - nonzero + nonzero * self.pruning_rate)
            thre = y[thre_index]
            pruned = 0
            print('Pruning threshold: {}'.format(thre))
            zero_flag = False
            for k, m in enumerate(self.model.modules()):
                if isinstance(m, nn.Conv2d):
                    weight_copy = m.weight.data.abs().clone()
                    mask = weight_copy.gt(thre).float().cuda()
                    pruned = pruned + mask.numel() - torch.sum(mask)
                    self.masks.append(mask)


                    print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                          format(k, mask.numel(), int(torch.sum(mask))))
            print(
                'Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned / total))

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
                        size = m.weight.data.numel()
                        total += size
                        temp = m.weight.data.abs().clone().gt(0).float().cuda()
                        nonzero = torch.sum(temp)
                        conv_weights = torch.zeros(size)
                        conv_weights = m.weight.data.view(-1).abs().clone()
                        y, _ = torch.sort(conv_weights)
                        thre_index = int(size - nonzero + nonzero * pruning_ratio)
                        thre = y[thre_index]

                        print('Pruning threshold: {}'.format(thre))
                        weight_copy = m.weight.data.abs().clone()
                        mask = weight_copy.gt(thre).float().cuda()
                        pruned = pruned + mask.numel() - torch.sum(mask)
                        self.masks.append(mask)

                        if int(torch.sum(mask)) == 0:
                            zero_flag = True
                        print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                              format(k, mask.numel(), int(torch.sum(mask))))
                    else:
                        print("layer {} is not Conv2d \n {}".format(k, m))
                    index += 1
                    if index >= len(self.target_layers):
                        break
                    layer_id, pruning_ratio = self.target_layers[index]

            print(
                'Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned / total))


    def zero_params(self, masks=None):
        """ Apply the masks, ie, zero out the params """
        masks = masks if masks is not None else self.masks

        layer = 0
        if self.layer_wise:
            layer_id, _ = self.target_layers.pop(0)
            for k, m in enumerate(self.model.modules()):
                if isinstance(m, nn.Conv2d) and k == layer_id:

                    m.weight.data.mul_(masks[layer])
                    layer += 1

                if self.target_layers:
                    layer_id, pruning_ratio = self.target_layers.pop(0)
                else:
                    break
        if self.global_wise:
            for k, m in enumerate(self.model.modules()):
                if isinstance(m, nn.Conv2d):

                    m.weight.data.mul_(masks[layer])
                    layer += 1



    ##############
# Global
# TEST RESULTS PRUNE ONCE
# {'Accuracy': 93.86}
# --------------------------------------------------------------------------------
# Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:08<00:00,  4.48it/s]
# Pruning threshold: 0.0008240019669756293
# layer index: 3   total params: 1728      remaining params: 1675
# layer index: 9   total params: 4096      remaining params: 3736
# layer index: 11          total params: 36864     remaining params: 31902
# layer index: 13          total params: 16384     remaining params: 13738
# layer index: 17          total params: 16384     remaining params: 14223
# layer index: 20          total params: 16384     remaining params: 13597
# layer index: 22          total params: 36864     remaining params: 30428
# layer index: 24          total params: 16384     remaining params: 11594
# layer index: 28          total params: 16384     remaining params: 13974
# layer index: 30          total params: 36864     remaining params: 30379
# layer index: 32          total params: 16384     remaining params: 12035
# layer index: 37          total params: 32768     remaining params: 28680
# layer index: 39          total params: 147456    remaining params: 118012
# layer index: 41          total params: 65536     remaining params: 53735
# layer index: 45          total params: 131072    remaining params: 103469
# layer index: 48          total params: 65536     remaining params: 49440
# layer index: 50          total params: 147456    remaining params: 115739
# layer index: 52          total params: 65536     remaining params: 52188
# layer index: 56          total params: 65536     remaining params: 53794
# layer index: 58          total params: 147456    remaining params: 120012
# layer index: 60          total params: 65536     remaining params: 50511
# layer index: 64          total params: 65536     remaining params: 54813
# layer index: 66          total params: 147456    remaining params: 117831
# layer index: 68          total params: 65536     remaining params: 44379
# layer index: 73          total params: 131072    remaining params: 106380
# layer index: 75          total params: 589824    remaining params: 388396
# layer index: 77          total params: 262144    remaining params: 163937
# layer index: 81          total params: 524288    remaining params: 228868
# layer index: 84          total params: 262144    remaining params: 126663
# layer index: 86          total params: 589824    remaining params: 303636
# layer index: 88          total params: 262144    remaining params: 90407
# layer index: 92          total params: 262144    remaining params: 124155
# layer index: 94          total params: 589824    remaining params: 256731
# layer index: 96          total params: 262144    remaining params: 54637
# layer index: 100         total params: 262144    remaining params: 95075
# layer index: 102         total params: 589824    remaining params: 130660
# layer index: 104         total params: 262144    remaining params: 31608
# layer index: 108         total params: 262144    remaining params: 92847
# layer index: 110         total params: 589824    remaining params: 115964
# layer index: 112         total params: 262144    remaining params: 37704
# layer index: 116         total params: 262144    remaining params: 90616
# layer index: 118         total params: 589824    remaining params: 141961
# layer index: 120         total params: 262144    remaining params: 63762
# layer index: 125         total params: 524288    remaining params: 78192
# layer index: 127         total params: 2359296   remaining params: 112271
# layer index: 129         total params: 1048576   remaining params: 148802
# layer index: 133         total params: 2097152   remaining params: 181095
# layer index: 136         total params: 1048576   remaining params: 23417
# layer index: 138         total params: 2359296   remaining params: 75918
# layer index: 140         total params: 1048576   remaining params: 108701
# layer index: 144         total params: 1048576   remaining params: 13483
# layer index: 146         total params: 2359296   remaining params: 71984
# layer index: 148         total params: 1048576   remaining params: 91692
# Total conv params: 23447232, Pruned conv params: 18757786.0, Pruned ratio: 0.800000011920929
# Testing:  98%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎    | 39/40 [00:07<00:00, 11.50it/s]-
# -------------------------------------------------------------------------------
# TEST RESULTS
# {'Accuracy': 93.41}
# --------------------------------------------------------------------------------
# Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:07<00:00,  5.26it/s]
#


# Local
# layers = [(120, 0.9), (125, 0.9), (138, 0.9), (140, 0.95)]
# TEST RESULTS
# {'Accuracy': 93.86}
# --------------------------------------------------------------------------------
# Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:08<00:00,  4.49it/s]
# layer wise pruning
# [(120, 0.9), (125, 0.9), (138, 0.9), (140, 0.95)]
# Pruning threshold: 0.0017784038791432977
# layer index: 120         total params: 262144    remaining params: 26214
# Pruning threshold: 0.0010467255488038063
# layer index: 125         total params: 524288    remaining params: 52428
# Pruning threshold: 0.00045475305523723364
# layer index: 138         total params: 2359296   remaining params: 235929
# Pruning threshold: 0.0017091594636440277
# layer index: 140         total params: 1048576   remaining params: 52428
# Total conv params: 4194304, Pruned conv params: 3827305.0, Pruned ratio: 0.9125006198883057
# Testing:  98%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎    | 39/40 [00:07<00:00, 11.64it/s]-
# -------------------------------------------------------------------------------
# TEST RESULTS
# {'Accuracy': 93.86}
# --------------------------------------------------------------------------------
# Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:07<00:00,  5.38it/s]

