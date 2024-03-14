from sklearn.model_selection import train_test_split
import torch.utils.data as torchdata
from torchvision.io import read_image
import torch
import sklearn
import numpy as np
import os


NUM_CLASSES = 6

label_map = {
    "buildings": 0,
    "forest": 1,
    "glacier": 2,
    "mountain": 3,
    "sea": 4,
    "street": 5
}
label_map_inv = {v: k for k, v in label_map.items()}



class IN3310Dataset(torchdata.Dataset):
    """
    Dataset for IN3310
    """

    def __init__(self, root_dir, transform=None):
        self.labels = []
        self.filenames = []
        self.root_dir = root_dir
        self.ending = ".jpg"
        self.transform = transform

        for _, dirs, files in os.walk(root_dir):
            for dir in dirs:
                for _, dirs, files in os.walk(os.path.join(root_dir, dir)):
                    for file in files:
                        if file.endswith(self.ending):
                            self.filenames.append(os.path.join(dir, file))
                            self.labels.append(label_map[dir])
            

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.filenames[index])
        img = read_image(img_path)
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label




def splitDataset(data, test_size, val_size, random_state, indicies=False):
    """
    Split the dataset into train, test and validation datasets
    """
    val_size = val_size / (1 - test_size) # Adjust val_size to be a fraction of the training set
    train_indices, test_indices = train_test_split(np.arange(len(data)), test_size=test_size, random_state=random_state, stratify=data.labels)
    train_indices, val_indices = train_test_split(train_indices, test_size=val_size, random_state=random_state, stratify=np.array(data.labels)[train_indices])

    train = torchdata.Subset(data, train_indices)
    test = torchdata.Subset(data, test_indices)
    val = torchdata.Subset(data, val_indices)
    if indicies:
        return train, test, val, train_indices, test_indices, val_indices
    return train, test, val




def verifyDisjoint(train, test, val):
    """
    Verify that the train, test and val datasets are disjoint
    """
    train = set([ train.dataset.filenames[i] for i in train.indices ])
    test = set([ test.dataset.filenames[i] for i in test.indices ])
    val = set([ val.dataset.filenames[i] for i in val.indices ])
    if len(train.intersection(test)) > 0:
        return False
    if len(train.intersection(val)) > 0:
        return False
    if len(test.intersection(val)) > 0:
        return False
    return True



def checkStratify(data, train, test, val):
    """
    Check that the stratify parameter works
    """
    train_labels = np.array(data.labels)[train.indices]
    test_labels = np.array(data.labels)[test.indices]
    val_labels = np.array(data.labels)[val.indices]
    train_labels, train_counts = np.unique(train_labels, return_counts=True)
    test_labels, test_counts = np.unique(test_labels, return_counts=True)
    val_labels, val_counts = np.unique(val_labels, return_counts=True)
    total_labels, total_counts = np.unique(data.labels, return_counts=True)
    print("Percentage of each class in train, test and val datasets:")
    print("{:<12}".format("Class:"), end="")
    for label in total_labels:
        print(f"{label_map_inv[label]:<12}", end="")
    print()
    print("{:<12}".format("Train:"), end="")
    for i in range(len(total_labels)):
        str_ = f"{train_counts[i] / total_counts[i]:.1%}"
        print(f"{str_:<12}", end="")
    print()
    print("{:<12}".format("Test:"), end="")
    for i in range(len(total_labels)):
        str_ = f"{test_counts[i] / total_counts[i]:.1%}"
        print(f"{str_:<12}", end="")
    print()
    print("{:<12}".format("Val:"), end="")
    for i in range(len(total_labels)):
        str_ = f"{val_counts[i] / total_counts[i]:.1%}"
        print(f"{str_:<12}", end="")
    print()





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



def test_epoch(model, epoch, data_loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    confusion_matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES).to(device)
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

            confusion_matrix = confusion_matrix + torch.bincount(NUM_CLASSES * labels + pred.view(-1), minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)

            if i % 5 == 0:
                print(f"Val poch: {epoch} | {(i + 1)/len(data_loader) * 100:.2f}% complete                ", end="\r")

    val_loss = running_loss / len(data_loader)
    logits = torch.cat(logits)
    true_labels = torch.cat(true_labels)
    return val_loss, confusion_matrix, logits, true_labels



def val_epoch(model, epoch, data_loader, loss_fn,device):
    model.eval()
    running_loss = 0.0
    confusion_matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES).to(device)
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

            confusion_matrix = confusion_matrix + torch.bincount(NUM_CLASSES * labels + pred.view(-1), minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)

            if i % 5 == 0:
                print(f"Val poch: {epoch} | {(i + 1)/len(data_loader) * 100:.2f}% complete                ", end="\r")

    val_loss = running_loss / len(data_loader)
    logits = torch.cat(logits)
    true_labels = torch.cat(true_labels)
    return val_loss, confusion_matrix, logits, true_labels



def train_epoch(model, epoch, data_loader, loss_fn, optimizer, device):

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

    train_loss = running_loss / len(data_loader)
    return train_loss
