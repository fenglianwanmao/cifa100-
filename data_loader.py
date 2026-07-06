
import pickle
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


def unpickle(file_path):
    with open(file_path, 'rb') as fo:
        return pickle.load(fo, encoding='latin1')


def load_raw_cifar100(data_dir):
    """Same as original: returns numpy arrays + label names."""
    meta = unpickle(os.path.join(data_dir, 'meta'))
    train = unpickle(os.path.join(data_dir, 'train'))
    test = unpickle(os.path.join(data_dir, 'test'))

    fine_names = meta['fine_label_names']
    coarse_names = meta['coarse_label_names']

    def _process(data):
        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N, H, W, C)
        return data.astype(np.float32) / 255.0

    x_train = _process(train['data'])
    x_test = _process(test['data'])

    y_train_fine = np.array(train['fine_labels'], dtype=np.int64)
    y_train_coarse = np.array(train['coarse_labels'], dtype=np.int64)
    y_test_fine = np.array(test['fine_labels'], dtype=np.int64)
    y_test_coarse = np.array(test['coarse_labels'], dtype=np.int64)

    return (x_train, y_train_fine, y_train_coarse,
            x_test, y_test_fine, y_test_coarse,
            fine_names, coarse_names)


class CIFAR100Dataset(Dataset):
    """PyTorch Dataset for CIFAR-100."""
    def __init__(self, images, labels, transform=None):
        self.images = images  # (N, H, W, C) float32 [0,1]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # HWC
        label = self.labels[idx]

        img = torch.from_numpy(img).permute(2, 0, 1)  # CHW

        if self.transform:
            img = self.transform(img)

        return img, label


def get_transforms(train=True):
    """Basic augmentations similar to original TF version."""
    if train:
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
        ])
    else:
        return T.Compose([])


def get_dataloaders(data_dir, label_type='coarse', batch_size=64, num_workers=2):
    
    (x_train, yf_tr, yc_tr, x_test, yf_te, yc_te, fine_names, coarse_names) = load_raw_cifar100(data_dir)

    if label_type == 'fine':
        y_train, y_test = yf_tr, yf_te
        class_names = fine_names
        num_classes = 100
    else:
        y_train, y_test = yc_tr, yc_te
        class_names = coarse_names
        num_classes = 20

    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    train_ds = CIFAR100Dataset(x_train, y_train, transform=train_transform)
    test_ds = CIFAR100Dataset(x_test, y_test, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, num_classes, class_names


if __name__ == "__main__":
    # Quick test
    train_loader, test_loader, n_classes, names = get_dataloaders(
        '../cifar-100-python', label_type='coarse', batch_size=8
    )
    print(f"Num classes: {n_classes}")
    print(f"Sample class names: {names[:5]}")
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, labels: {labels}")
    print("DataLoader OK (PyTorch version)")