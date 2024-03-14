from sklearn.model_selection import train_test_split
import torch.utils.data as torchdata
from torchvision.io import read_image
import numpy as np
import os



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




def splitDataset(data, test_size, val_size, random_state):
    """
    Split the dataset into train, test and validation datasets
    """
    val_size = val_size / (1 - test_size) # Adjust val_size to be a fraction of the training set
    train_indices, test_indices = train_test_split(np.arange(len(data)), test_size=test_size, random_state=random_state, stratify=data.labels)
    train_indices, val_indices = train_test_split(train_indices, test_size=val_size, random_state=random_state, stratify=np.array(data.labels)[train_indices])

    train = torchdata.Subset(data, train_indices)
    test = torchdata.Subset(data, test_indices)
    val = torchdata.Subset(data, val_indices)
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
