import torch.nn as nn

def mlp(sizes, activation):
    layers = []
    for j in range(len(sizes)-1):
        layers += [nn.Linear(sizes[j], sizes[j+1]), activation()]
    return nn.Sequential(*layers)