from dataset import IN3310Dataset, splitDataset, verifyDisjoint, checkStratify
import torch.utils.data as torchdata
import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import sklearn
import numpy as np
seed = 42
torch.manual_seed(seed)

batch_size = 16





def accuracy(confusion_matrix):
    confusion_matrix = confusion_matrix.cpu()
    return confusion_matrix.diag().sum() / confusion_matrix.sum()

def accuracy_per_class(confusion_matrix):
    confusion_matrix = confusion_matrix.cpu()
    return confusion_matrix.diag() / confusion_matrix.sum(dim=1)


def precision_recall_curve(y_true, y_score, batch_size):
    y_true = y_true.cpu().numpy()
    y_score = y_score.detach().cpu().numpy()
    binlables = sklearn.preprocessing.label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5])
    precision = np.zeros((6,batch_size+1))
    recall = np.zeros((6,batch_size+1))
    for i in range(6):
        p, r, _ = sklearn.metrics.precision_recall_curve(binlables[:, i], y_score[:, i])
        precision[i] = p
        recall[i] = r
    return precision, recall



def AP(y_true, y_score):
    y_true = y_true.cpu().numpy()
    y_score = y_score.detach().cpu().numpy()
    ap = sklearn.metrics.average_precision_score(y_true, y_score, average=None)
    return ap




# inspired by weekly exercise 3 solution
def train_epoch(model, epoch, data_loader, loss_fn, optimizer):

    model.train()
    running_loss = 0.0

    for i, batch in enumerate(data_loader):
        X, y = batch
        images, labels = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(images)
        loss = loss_fn(y_pred, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 5 == 0:
            print(f"Train epoch: {epoch} | {(i + 1)/len(data_loader) * 100:.2f}% complete                 ", end="\r")

    train_loss = running_loss / len(train_loader)
    return train_loss



def val_epoch(model, epoch, data_loader, loss_fn):
    model.eval()
    running_loss = 0.0
    confusion_matrix = torch.zeros(num_classes, num_classes).to(device)
    logits = []
    true_labels = []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            X, y = batch
            images, labels = X.to(device), y.to(device)
            y_pred = model(images)
            loss = loss_fn(y_pred, labels)
            running_loss += loss.item()

            pred = y_pred.argmax(dim=1)
            logits.append(y_pred)
            true_labels.append(labels)

            confusion_matrix = confusion_matrix + torch.bincount(num_classes * labels + pred.view(-1), minlength=num_classes ** 2).view(num_classes, num_classes)

            if i % 5 == 0:
                print(f"Val poch: {epoch} | {(i + 1)/len(data_loader) * 100:.2f}% complete                ", end="\r")

    val_loss = running_loss / len(data_loader)
    logits = torch.cat(logits)
    true_labels = torch.cat(true_labels)
    return val_loss, confusion_matrix, logits, true_labels



# ****************************************************
# ************ Main **********************************
# ****************************************************



weights = ResNet18_Weights.DEFAULT
"""
transformation required by the pretrained model (ResNet18)
https://pytorch.org/hub/pytorch_vision_resnet/
"""
preprocess = weights.transforms()
invPreprocess = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]), transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])])

data = IN3310Dataset("mandatory1_data", transform=preprocess)
train, test, val = splitDataset(data, 0.15, 0.15, seed)
checkStratify(data, train, test, val)
assert verifyDisjoint(train, test, val)

train_loader = torchdata.DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = torchdata.DataLoader(test, batch_size=batch_size, shuffle=True)
val_loader = torchdata.DataLoader(val, batch_size=batch_size, shuffle=True)

num_classes = 6
model = resnet18(weights=weights)

"""
freeze all layers
we want to keep most of the model as it is, but finetune the last layer(s)
"""


for param in model.parameters():
    if isinstance(param, torch.nn.Conv2d):
        param.requires_grad = False

"""
fc (fully connected) is the last layer of the model
from the source code:
    self.fc = nn.Linear(512 * block.expansion, num_classes)

we can change this to fit our needs
alternatively, we can also add a new layer on top of the existing one
"""

num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)

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

epochs = 3
train_loss = np.zeros(epochs)
val_loss = np.zeros(epochs)
val_accuracy = np.zeros(epochs)
val_mAP = np.zeros(epochs)


for i in range(epochs):
    train_l = train_epoch(model, i, train_loader, loss_fn, optimizer)
    val_l, confusion_matrix, logits, true_labels = val_epoch(model, i, val_loader, loss_fn)
    val_accuracy[i] = accuracy(confusion_matrix)
    val_mAP[i] = AP(true_labels, logits).mean()
    train_loss[i] = train_l
    val_loss[i] = val_l
    print(f"Epoch: {i} | Validation Loss: {val_l:.5f} | Validation Accuracy: {val_accuracy[i]:.2f}%")
    # print(accuracy_per_class(confusion_matrix))
    # print(AP(true_labels, logits)) 

# ****************************************************
# ************ Evaluation and plotting ***************
# ****************************************************

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

# plt.savefig("training_only_freeze_conv.png", dpi=150, bbox_inches="tight")
plt.show()





"""
Evaluate the model on the test set dataloader
"""
test_loss, confusion_matrix, logits, true_labels = val_epoch(model, 0, test_loader, loss_fn)
test_accuracy = accuracy(confusion_matrix)
test_mAP = AP(true_labels, logits).mean()
softmax_test = torch.nn.Softmax(dim=1)(logits)

# write the results to a file
with open("results.txt", "w") as file:
    file.write(f"Test Loss: {test_loss:.5f} | Test Accuracy: {test_accuracy:.2f}% | Test mAP: {test_mAP:.2f}%")
    file.write(f"Sofmax output: {softmax_test}")





# ****************************************************
# ************ Feature Map Visualization *************
# ****************************************************


"""
choosing two feature maps from the first layer, two from the second layer, and one from the third layer
"""
activations = {}

def get_activation(name):
    def hook_fn(module, input, output):
        activations[name] = output
    return hook_fn

"""
Register hooks
"""
model.layer1[0].relu.register_forward_hook(get_activation("layer1.0.relu"))
model.layer1[1].relu.register_forward_hook(get_activation("layer1.1.relu"))
model.layer2[0].relu.register_forward_hook(get_activation("layer2.0.relu"))
model.layer2[1].relu.register_forward_hook(get_activation("layer2.1.relu"))
model.layer3[0].relu.register_forward_hook(get_activation("layer3.0.relu"))


featurestats = {}

with torch.no_grad():
    for i in range(13): # 13 batches of 16 images gives us 208 images 
        X,y = next(iter(train_loader))
        images, lables = X.to(device), y.to(device)
        model(images)
        for key in activations:
            non_zero = torch.count_nonzero(activations[key])
            if key in featurestats:
                featurestats[key] += non_zero / torch.prod(torch.tensor(activations[key].shape))
            else:
                featurestats[key] = non_zero / torch.prod(torch.tensor(activations[key].shape))



for key in featurestats:
    featurestats[key] /= 13

featurestats = torch.tensor([featurestats[key] for key in featurestats]).numpy()

# write the results to a file
with open("featurestats.txt", "w") as file:
    file.write(f"Feature map statistics: {featurestats}")




with torch.no_grad():
    X,y = next(iter(train_loader))
    images, lables = X.to(device), y.to(device)
    model(images)


"""
Plot the feature maps from 16 channels/filters in each layer
"""
X = invPreprocess(X[1])
fig, ax = plt.subplots(1, 1)
ax.imshow(X.permute(1, 2, 0).cpu().numpy())
ax.axis("off")
plt.title("Original image")
# plt.savefig("original-image.png", dpi=150, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(5, 16, figsize=(16, 8))
for j in range(16):
    ax[0, j].imshow(activations[f"layer1.0.relu"][1, j].cpu().numpy())
    ax[0, j].axis("off")
    ax[1, j].imshow(activations[f"layer1.1.relu"][1, j].cpu().numpy())
    ax[1, j].axis("off")
    ax[2, j].imshow(activations[f"layer2.0.relu"][1, j].cpu().numpy())
    ax[2, j].axis("off")
    ax[3, j].imshow(activations[f"layer2.1.relu"][1, j].cpu().numpy())
    ax[3, j].axis("off")
    ax[4, j].imshow(activations[f"layer3.0.relu"][1, j].cpu().numpy())
    ax[4, j].axis("off")
fig.suptitle("Feature maps from the first 3 layers")
fig.tight_layout(
    pad=0.1, w_pad=0.1, h_pad=0.1
)
# plt.savefig("featuremaps.png", dpi=150, bbox_inches="tight")
plt.show()
