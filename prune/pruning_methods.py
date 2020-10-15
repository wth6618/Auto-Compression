import torch
import torch.nn as nn
import prune

def unstructured_pruner(net, method, ratio = 0.7):
    assert method in ["Magnitude", "Vector", "Sensitivity"], "Method Not Avaliable Now!"
    net = net.to("cuda")
    if method == "Magnitude":
        pruner = prune.Unstructured(net, pruning_rate=ratio, global_wise=True)
        pruner.step()
        pruner.zero_params()
        return net

    elif method == "Vector":
        pruner = prune.VectorPruning(net, pruning_rate=ratio, global_wise=True)
        pruner.step()
        pruner.zero_params()
        return net

    elif method == "Sensitivity":
        pruner = prune.Sensitivity_Pruning(net)
        pruner.step()
        pruner.zero_params()
        return net









