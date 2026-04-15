from pathlib import Path
from typing import Optional, Callable
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import gdown
import zipfile
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn

import sklearn

##########################################
### Constants and utilities for Part 1 ###
##########################################


BASE_DATA_PATH = Path(".") / "data"
ZIP_PATH = BASE_DATA_PATH / "part1.zip"
DATA_PATH = BASE_DATA_PATH / "part1"


FRUIT_CLASS_NAMES = [
    "granny_smith",
    "strawberry",
    "orange",
    "lemon",
    "fig",
    "pineapple",
    "banana",
    "jackfruit",
    "custard_apple",
    "pomegranate",
]


def create_feature_extractor(
    model_name: str, pretrained_cfg: str, device: str = "cuda:0"
):
    model = timm.create_model(
        model_name, pretrained=True, pretrained_cfg=pretrained_cfg
    )
    model.transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    model.head = torch.nn.Identity()
    return model.eval().to(device)


datas = {}


def download_data():
    BASE_DATA_PATH.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        file_id = "1qNFjQIBck90I41aiJMpZR70NZRHQtEqE"
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            output=str(ZIP_PATH),
            quiet=True,
        )

    if not DATA_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(str(BASE_DATA_PATH))


def load_data(split_name: str):
    download_data()
    images = np.load(DATA_PATH / f"{split_name}_images.npy")
    labels = np.load(DATA_PATH / f"{split_name}_labels.npy")
    return images, labels


def get_data(split_name: str):
    if split_name not in datas:
        datas[split_name] = load_data(split_name)
    return datas[split_name]


class FruitDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split_name: str,
        transform: Optional[Callable] = None,
    ):
        self.images, self.labels = get_data(split_name)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def num_classes(self):
        return np.max(self.labels) + 1


def visualize_samples(
    split_name: str,
    num_rows: int = 2,
    figsize: tuple[int, int] = (10, 5),
    seed: int = 0,
):
    dataset = FruitDataset(split_name)
    random = np.random.RandomState(seed)
    _, axes = plt.subplots(num_rows, dataset.num_classes // num_rows, figsize=figsize)
    for class_idx in range(dataset.num_classes):
        row = class_idx // (dataset.num_classes // num_rows)
        col = class_idx % (dataset.num_classes // num_rows)
        class_indices = np.where(dataset.labels == class_idx)[0]
        random_idx = random.choice(class_indices)
        img, _ = dataset[random_idx]
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"{FRUIT_CLASS_NAMES[class_idx]}")
        axes[row, col].axis("off")
    plt.tight_layout()
    plt.show()


#########################################
### Part 1.1: Training a linear probe ###
#########################################



def get_features(
    split_name: str,
    feature_extractor: torch.nn.Module,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda:0",
):
    """Get the features for the given split. Hint: use torch.no_grad() to disable gradient computation.

    Args:
        split_name: The name of the split to get features from.
        feature_extractor: The feature extractor to use.
        batch_size: The batch size to use for loading data.
        num_workers: The number of workers to use for loading data.
        device: The device to use.

    Returns:
        features: A numpy array of shape (num_examples, num_features) containing the features.
        labels: A numpy array of shape (num_examples,) containing the labels (used to construct the FeaturesDataset).
        num_classes: The number of classes in the dataset (used to construct the FeaturesDataset).
    """
    dataset = FruitDataset(split_name, transform=feature_extractor.transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    all_features = []
    all_labels = []

    feature_extractor.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"Extracting features ({split_name})"):
            x = x.to(device)
            features = feature_extractor(x)
            if features.ndim > 2:
                features = features.flatten(start_dim=1)
            all_features.append(features.cpu().numpy())
            all_labels.append(y.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels, dataset.num_classes



class FeaturesDataset(torch.utils.data.Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, num_classes: int):
        self.features = features
        self.labels = labels
        self.num_classes = num_classes

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

    @classmethod
    def create(cls, split_name: str, feature_extractor: torch.nn.Module, **kwargs):
        features, labels, num_classes = get_features(
            split_name, feature_extractor, **kwargs
        )
        return cls(features, labels, num_classes)


def train_linear_probe(
    features_dataset: FeaturesDataset,
    num_epochs: int = 32,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    num_workers: int = 4,
    device: str = "cuda:0",
):
    """Train a linear probe on the training split of the fruit dataset. This should use the Adam optimizer (with hyperparameters as specified in the default arguments).

    Args:
        features_dataset: A FeaturesDataset object to train on.
        num_epochs: The number of epochs to train for.
        batch_size: The batch size to use for training.
        learning_rate: The learning rate to use for training.
        weight_decay: The weight decay to use for training.
        num_workers: The number of workers to use for data loading.
        device: The device to use for training.

    Returns:
        linear_probe: A torch.nn.Linear object representing the trained linear probe.
        epoch_losses: A list of length num_epochs containing the average loss over each epoch.
    """
    loader = DataLoader(
        features_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    num_features = features_dataset.features.shape[1]
    linear_probe = nn.Linear(num_features, features_dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(
        linear_probe.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    epoch_losses = []
    linear_probe.train()
    for _ in tqdm(range(num_epochs), desc="Training linear probe"):
        total_loss = 0.0
        total_examples = 0
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)

            optimizer.zero_grad()
            logits = linear_probe(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            batch_size_actual = y.shape[0]
            total_loss += loss.item() * batch_size_actual
            total_examples += batch_size_actual

        epoch_losses.append(total_loss / total_examples)

    return linear_probe, epoch_losses




def evaluate_linear_probe(
    features_dataset: FeaturesDataset,
    linear_probe: torch.nn.Module,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda:0",
):
    linear_probe.eval()
    loader = torch.utils.data.DataLoader(
        features_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            outputs = linear_probe(x)
        preds = outputs.argmax(dim=1)
        total += len(y)
        correct += (preds == y).sum().item()

    accuracy = correct / total
    return accuracy


###############################################################
### Part 1.2: Analyzing representations out-of-distribution ###
###############################################################



def find_nearest_neighbors(
    query_features: np.ndarray, features: np.ndarray, k: int = 5
) -> float:
    """Find the k nearest neighbors of the given query features according to cosine similarity.

    Args:
        query_features: A numpy array of shape (num_features,) containing the features for one example.
        features: A numpy array of shape (num_examples, num_features) containing the features in which to find the nearest neighbors.

    Returns:
        indices: A numpy array of shape (k,) containing the indices of the k nearest neighbors of the query features in the features array.
        similarities: A numpy array of shape (k,) containing the similarities between the query features and the nearest neighbors.
    """
    query_norm = np.linalg.norm(query_features)
    feature_norms = np.linalg.norm(features, axis=1)
    similarities = features @ query_features / (
        np.maximum(feature_norms * query_norm, 1e-12)
    )

    nearest_indices = np.argsort(similarities)[-k:][::-1]
    return nearest_indices, similarities[nearest_indices]

def visualize_nearest_neighbors(photo_features, sketch_features, method_name, query_index, k=5):
    _, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 3))

    photo_dataset = FruitDataset("photo_val")
    query_image, query_label = photo_dataset[query_index]
    axes[0].imshow(query_image)
    axes[0].set_title(f"Query: {FRUIT_CLASS_NAMES[query_label]}")
    axes[0].axis('off')

    sketch_dataset = FruitDataset("sketch_val")
    nearest_indices, similarities = find_nearest_neighbors(photo_features[method_name][query_index], sketch_features[method_name], k=k)

    for i, (index, similarity) in enumerate(zip(nearest_indices, similarities)):
        sketch_image, sketch_label = sketch_dataset[index]

        axes[i+1].imshow(sketch_image)
        axes[i+1].set_title(f"{FRUIT_CLASS_NAMES[sketch_label]}\nSimilarity: {similarity:.3f}")
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.show()
