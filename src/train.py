import torch
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import matplotlib.pyplot as plt
from util import *


seed = 42
torch.manual_seed(seed)
batch_size = 16


weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()

data = IN3310Dataset("../mandatory1_data", transform=preprocess)
train_ind = torch.load("../train_indices.pt")
val_ind = torch.load("../val_indicies.pt")
train = torch.utils.data.Subset(data, train_ind)
val = torch.utils.data.Subset(data, val_ind)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)


model = resnet18(weights=weights)

"""
freeze all layers
we want to keep most of the model as it is, but finetune the last layer(s)
"""

for param in model.parameters():
    param.requires_grad = False

"""
fc (fully connected) is the last layer of the model
from the source code:
    self.fc = nn.Linear(512 * block.expansion, num_classes)

we can change this to fit our needs
alternatively, we can also add a new layer on top of the existing one
"""

num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# use the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")
print("Starting training...\n\n")

# ****************************************************
# ************ Training *******************************
# ****************************************************

epochs = 100
train_loss = np.zeros(epochs)
val_loss = np.zeros(epochs)
val_accuracy = np.zeros(epochs)
val_mAP = np.zeros(epochs)


for i in range(epochs):
    train_l = train_epoch(model, i, train_loader, loss_fn, optimizer, device)
    val_l, confusion_matrix, logits, true_labels = val_epoch(model, i, val_loader, loss_fn,device)
    val_accuracy[i] = accuracy(confusion_matrix)
    val_mAP[i] = AP(true_labels, logits).mean()
    train_loss[i] = train_l
    val_loss[i] = val_l
    print(f"Epoch: {i} | Validation Loss: {val_l:.5f} | Validation Accuracy: {val_accuracy[i]:.2f}%")
    # print(accuracy_per_class(confusion_matrix))
    # print(AP(true_labels, logits)) 


# save the model
torch.save(model.state_dict(), "../model.pt")

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(train_loss, label="Train Loss")
ax[0].plot(val_loss, label="Validation Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].legend()

ax[1].plot(val_accuracy, label="Validation Accuracy")
ax[1].plot(val_mAP, label="Validation mAP")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].legend()

plt.savefig("../training_only_freeze_conv.png", dpi=150, bbox_inches="tight")
# plt.show()
