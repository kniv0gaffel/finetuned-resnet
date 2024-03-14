from util import IN3310Dataset, splitDataset, checkStratify, verifyDisjoint
import torch

seed = 42
torch.manual_seed(seed)

data = IN3310Dataset("../mandatory1_data", transform=None)
train, test, val, train_ind, val_ind, test_ind = splitDataset(data, 0.15, 0.15, seed, indicies=True)
checkStratify(data, train, test, val)
assert verifyDisjoint(train, test, val)

# save the indices for later use to ensure reproducibility
torch.save(train_ind, "../train_indices.pt")
torch.save(val_ind, "../val_indicies.pt")
torch.save(test_ind, "../test_indicies.pt")
