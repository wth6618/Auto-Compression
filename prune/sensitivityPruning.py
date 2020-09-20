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


class Sensitivity_Pruning():
    """ Magnitude pruning with an optimizer-like interface  """

    def __init__(self, model, sensitivity=0.86):
        """ Init pruning method """
        self.sensitivity = float(sensitivity)

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


        total = 0
        pruned = 0
        index = 0
        for k, m in enumerate(self.model.modules()):
            if isinstance(m, nn.Conv2d):
                size = m.weight.data.numel()
                total += size
                conv_weights = torch.zeros(size)
                conv_weights = m.weight.data.view(-1).abs().clone()

                thre = torch.abs(torch.std(conv_weights)) * self.sensitivity

                print('Pruning threshold: {}'.format(thre))
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(thre).float().cuda()
                pruned = pruned + mask.numel() - torch.sum(mask)
                self.masks.append(mask)

                if int(torch.sum(mask)) == 0:
                    zero_flag = True
                print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                      format(k, mask.numel(), int(torch.sum(mask))))
        print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned / total))


            # for i, (m, p) in enumerate(zip(self.masks, self.model.modules())):
            #     # Compute cutoff
            #     if isinstance()
            #     flat_params = p[m == 1].view(-1)
            #     cutoff = cut(self.pruning_rate, flat_params)
            #     # Update mask
            #     new_mask = torch.where(torch.abs(p) < cutoff,
            #                            torch.zeros_like(p), m)
            #     self.masks[i] = new_mask

    def zero_params(self, masks=None):
        """ Apply the masks, ie, zero out the params """
        masks = masks if masks is not None else self.masks
        # for m, p in zip(masks, self.params):
        #     p.data = m * p.data
        layer = 0

        for k, m in enumerate(self.model.modules()):
            if isinstance(m, nn.Conv2d):
                # print('layer {}, mask: {}'.format(m, masks[layer]))
                m.weight.data.mul_(masks[layer])
                layer += 1

    ##############

# python -W ignore cifar10_test.py --classifier resnet50 --gpu 0

# TEST RESULTS
# {'Accuracy': 93.86}
# --------------------------------------------------------------------------------
# Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:08<00:00,  4.46it/s]
# Pruning threshold: 0.02069055661559105
# layer index: 3   total params: 1728      remaining params: 846
# Pruning threshold: 0.007059002760797739
# layer index: 9   total params: 4096      remaining params: 1769
# Pruning threshold: 0.003195801516994834
# layer index: 11          total params: 36864     remaining params: 19500
# Pruning threshold: 0.004334835335612297
# layer index: 13          total params: 16384     remaining params: 7383
# Pruning threshold: 0.005572755355387926
# layer index: 17          total params: 16384     remaining params: 5940
# Pruning threshold: 0.002897511934861541
# layer index: 20          total params: 16384     remaining params: 7820
# Pruning threshold: 0.002660228870809078
# layer index: 22          total params: 36864     remaining params: 18230
# Pruning threshold: 0.0034507685340940952
# layer index: 24          total params: 16384     remaining params: 5678
# Pruning threshold: 0.0030843238346278667
# layer index: 28          total params: 16384     remaining params: 8226
# Pruning threshold: 0.002838526852428913
# layer index: 30          total params: 36864     remaining params: 17379
# Pruning threshold: 0.0033626260701566935
# layer index: 32          total params: 16384     remaining params: 5881
# Pruning threshold: 0.0034442059695720673
# layer index: 37          total params: 32768     remaining params: 16700
# Pruning threshold: 0.002013625344261527
# layer index: 39          total params: 147456    remaining params: 80123
# Pruning threshold: 0.002607761649414897
# layer index: 41          total params: 65536     remaining params: 32923
# Pruning threshold: 0.0021611901465803385
# layer index: 45          total params: 131072    remaining params: 65229
# Pruning threshold: 0.0017168492777273059
# layer index: 48          total params: 65536     remaining params: 34254
# Pruning threshold: 0.0018075504340231419
# layer index: 50          total params: 147456    remaining params: 82059
# Pruning threshold: 0.002418027725070715
# layer index: 52          total params: 65536     remaining params: 32405
# Pruning threshold: 0.002170118736103177
# layer index: 56          total params: 65536     remaining params: 36056
# Pruning threshold: 0.002039325423538685
# layer index: 58          total params: 147456    remaining params: 82450
# Pruning threshold: 0.002330129500478506
# layer index: 60          total params: 65536     remaining params: 30459
# Pruning threshold: 0.0023748516105115414
# layer index: 64          total params: 65536     remaining params: 36204
# Pruning threshold: 0.001995789585635066
# layer index: 66          total params: 147456    remaining params: 79812
# Pruning threshold: 0.0020318212918937206
# layer index: 68          total params: 65536     remaining params: 25238
# Pruning threshold: 0.0021556520368903875
# layer index: 73          total params: 131072    remaining params: 70279
# Pruning threshold: 0.0011491248151287436
# layer index: 75          total params: 589824    remaining params: 318880
# Pruning threshold: 0.0012661677319556475
# layer index: 77          total params: 262144    remaining params: 123007
# Pruning threshold: 0.0008209579973481596
# layer index: 81          total params: 524288    remaining params: 229582
# Pruning threshold: 0.0008803845848888159
# layer index: 84          total params: 262144    remaining params: 119755
# Pruning threshold: 0.0009385316516272724
# layer index: 86          total params: 589824    remaining params: 274940
# Pruning threshold: 0.001115654595196247
# layer index: 88          total params: 262144    remaining params: 68756
# Pruning threshold: 0.0008878072840161622
# layer index: 92          total params: 262144    remaining params: 116356
# Pruning threshold: 0.0008543732692487538
# layer index: 94          total params: 589824    remaining params: 248689
# Pruning threshold: 0.0009101041941903532
# layer index: 96          total params: 262144    remaining params: 49586
# Pruning threshold: 0.0007653863867744803
# layer index: 100         total params: 262144    remaining params: 102476
# Pruning threshold: 0.0006254000472836196
# layer index: 102         total params: 589824    remaining params: 176657
# Pruning threshold: 0.0006885250331833959
# layer index: 104         total params: 262144    remaining params: 38330
# Pruning threshold: 0.0008250783430412412
# layer index: 108         total params: 262144    remaining params: 92698
# Pruning threshold: 0.0005642918986268342
# layer index: 110         total params: 589824    remaining params: 180026
# Pruning threshold: 0.0007075155153870583
# layer index: 112         total params: 262144    remaining params: 44106
# Pruning threshold: 0.0008896150393411517
# layer index: 116         total params: 262144    remaining params: 83694
# Pruning threshold: 0.0005424447590485215
# layer index: 118         total params: 589824    remaining params: 225558
# Pruning threshold: 0.0009275551419705153
# layer index: 120         total params: 262144    remaining params: 57136
# Pruning threshold: 0.0004982816753908992
# layer index: 125         total params: 524288    remaining params: 151686
# Pruning threshold: 0.00023616288672201335
# layer index: 127         total params: 2359296   remaining params: 1005110
# Pruning threshold: 0.0006122707854956388
# layer index: 129         total params: 1048576   remaining params: 224591
# Pruning threshold: 0.00043476460268720984
# layer index: 133         total params: 2097152   remaining params: 426350
# Pruning threshold: 0.00018397351959720254
# layer index: 136         total params: 1048576   remaining params: 361722
# Pruning threshold: 0.00022692081984132528
# layer index: 138         total params: 2359296   remaining params: 644042
# Pruning threshold: 0.0006257454515434802
# layer index: 140         total params: 1048576   remaining params: 145949
# Pruning threshold: 0.00015655440802220255
# layer index: 144         total params: 1048576   remaining params: 365449
# Pruning threshold: 0.00023030603188090026
# layer index: 146         total params: 2359296   remaining params: 552635
# Pruning threshold: 0.0006526833167299628
# layer index: 148         total params: 1048576   remaining params: 113498
# Total conv params: 23447232, Pruned conv params: 16103125.0, Pruned ratio: 0.6867814660072327
# Testing:  98%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎    | 39/40 [00:06<00:00, 11.65it/s]-
# -------------------------------------------------------------------------------
# TEST RESULTS
# {'Accuracy': 89.77}
# --------------------------------------------------------------------------------
# Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:07<00:00,  5.39it/s]
#
