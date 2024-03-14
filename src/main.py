import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import numpy as np
from util import *

seed = 42
torch.manual_seed(seed)
batch_size = 16



invPreprocess = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]), transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])])

weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()

data = IN3310Dataset("../mandatory1_data", transform=preprocess)
test_ind = torch.load("../test_indicies.pt")
test = torch.utils.data.Subset(data, test_ind)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

model = resnet18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)
# load the weights from the previous training
model.load_state_dict(torch.load("../model.pt"))

loss_fn = torch.nn.CrossEntropyLoss()
# use the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


"""
Evaluate the model on the test set dataloader
"""
test_loss, confusion_matrix, logits, true_labels = test_epoch(model, 0, test_loader, loss_fn, device)
test_accuracy = accuracy(confusion_matrix)
test_mAP = AP(true_labels, logits).mean()
softmax_test = torch.nn.Softmax(dim=1)(logits).cpu().numpy()

# write the results to a file
with open("../results.txt", "w") as file:
    file.write(f"Test Loss: {test_loss:.5f} | Test Accuracy: {test_accuracy:.2f}% | Test mAP: {test_mAP:.2f}%\n\n")

np.savetxt("../softmx_test.txt", softmax_test, fmt="%.3f", delimiter=",")

# best and worst scores for the second class
true_labels = true_labels.cpu().numpy()

second_class_indices = np.where(true_labels == 1)[0]
second_class_images = np.array([invPreprocess(test[i][0]).permute(1, 2, 0) for i in second_class_indices])
softmax_second_class = softmax_test[second_class_indices, 1]
ten_worst_second_class = np.argsort(softmax_second_class)[:10]
ten_best_second_class = np.argsort(softmax_second_class)[-10:]

sixth_class_indices = np.where(true_labels == 5)[0]
sixth_class_images = np.array([invPreprocess(test[i][0]).permute(1, 2, 0) for i in sixth_class_indices])
softmax_sixth_class = softmax_test[sixth_class_indices, 5]
ten_worst_sixth_class = np.argsort(softmax_sixth_class)[:10]
ten_best_sixth_class = np.argsort(softmax_sixth_class)[-10:]

first_class_indices = np.where(true_labels == 0)[0]
first_class_images = np.array([invPreprocess(test[i][0]).permute(1, 2, 0) for i in first_class_indices])
softmax_first_class = softmax_test[first_class_indices, 0]
ten_worst_first_class = np.argsort(softmax_first_class)[:10]
ten_best_first_class = np.argsort(softmax_first_class)[-10:]

ten_worst_second_class_imgs = second_class_images[ten_worst_second_class]
ten_best_second_class_imgs = second_class_images[ten_best_second_class]

fig, ax = plt.subplots(2, 10, figsize=(20, 4))
for i in range(10):
    ax[0, i].imshow(ten_worst_second_class_imgs[i])
    ax[0, i].title.set_text(f"{softmax_second_class[ten_worst_second_class[i]]:.3f}")
    ax[0, i].axis("off")
    ax[1, i].imshow(ten_best_second_class_imgs[i])
    ax[1, i].title.set_text(f"{softmax_second_class[ten_best_second_class[i]]:.3f}")
    ax[1, i].axis("off")

plt.savefig("../best_worst_second_class.png", dpi=150, bbox_inches="tight")

ten_worst_sixth_class_imgs = sixth_class_images[ten_worst_sixth_class]
ten_best_sixth_class_imgs = sixth_class_images[ten_best_sixth_class]

fig, ax = plt.subplots(2, 10, figsize=(20, 4))
for i in range(10):
    ax[0, i].imshow(ten_worst_sixth_class_imgs[i])
    ax[0, i].title.set_text(f"{softmax_sixth_class[ten_worst_sixth_class[i]]:.3f}")
    ax[0, i].axis("off")
    ax[1, i].imshow(ten_best_sixth_class_imgs[i])
    ax[1, i].title.set_text(f"{softmax_sixth_class[ten_best_sixth_class[i]]:.3f}")
    ax[1, i].axis("off")

plt.savefig("../best_worst_sixth_class.png", dpi=150, bbox_inches="tight")

ten_worst_first_class_imgs = first_class_images[ten_worst_first_class]
ten_best_first_class_imgs = first_class_images[ten_best_first_class]

fig, ax = plt.subplots(2, 10, figsize=(20, 4))
for i in range(10):
    ax[0, i].imshow(ten_worst_first_class_imgs[i])
    ax[0, i].title.set_text(f"{softmax_first_class[ten_worst_first_class[i]]:.3f}")
    ax[0, i].axis("off")
    ax[1, i].imshow(ten_best_first_class_imgs[i])
    ax[1, i].title.set_text(f"{softmax_first_class[ten_best_first_class[i]]:.3f}")
    ax[1, i].axis("off")

plt.savefig("../best_worst_first_class.png", dpi=150, bbox_inches="tight")



# test again to see if the results are the same
test_loss, confusion_matrix, logits, true_labels = test_epoch(model, 0, test_loader, loss_fn, device)
softmax_test2 = torch.nn.Softmax(dim=1)(logits).cpu().numpy()

# compare the softmax values
print()
print("Are the softmax values from two predictions on the test set the same?")
print(np.allclose(softmax_test, softmax_test2, atol=1e-6))



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
        X,y = next(iter(test_loader))
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
with open("../featurestats.txt", "w") as file:
    file.write(f"Feature map statistics: {featurestats}")




with torch.no_grad():
    X,y = next(iter(test_loader))
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
plt.savefig("../original-image.png", dpi=150, bbox_inches="tight")
# plt.show()

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
plt.savefig("../featuremaps.png", dpi=150, bbox_inches="tight")
# plt.show()
