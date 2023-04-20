import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

model = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(), 
    nn.Linear(10, 2)
)

prune.random_structured(model, amount=0.5, dim=1)

