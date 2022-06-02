import argparse
import json
import random
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from model import EfficientNet


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def get_transforms(gamma: float, phi: float) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(round(224 * pow(gamma, phi))),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )


def get_dataloaders(
    dataset: str, batch_size: int, gamma: float, phi: float
) -> list:
    transform = get_transforms(gamma, phi)
    if dataset == "ImageNet":
        train_set = torchvision.datasets.ImageNet(
            root='./data', train=True, download=True, transform=transform
        )
        test_set = torchvision.datasets.ImageNet(
            root='./data', train=False, download=True, transform=transform
        )
    else:
        train_set = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        test_set = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader, train_set.classes


def validate(
    model: nn.Module,
    loader: DataLoader,
    classes: list,
    device: torch.device,
    result_file: str,
    console: bool = False,
    model_result_folder: str = "./",
) -> None:
    correct_pred = {data_class: 0 for data_class in classes}
    total_pred = {data_class: 0 for data_class in classes}
    label2class = {idx: data_class for idx, data_class in enumerate(classes)}
    model.eval()
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.cpu().detach().numpy()
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[label2class[int(label)]] += 1
                total_pred[label2class[int(label)]] += 1
    if console:
        for x in classes:
            print(
                f"Validation accuracy for class {x} is "
                f"{correct_pred[x] * 100.0 / total_pred[x]:.3f}",
                flush=True
            )
        return None
    with open(result_file, "w") as f:
        for x in classes:
            print(
                f"Validation accuracy for class {x} is "
                f"{correct_pred[x] * 100.0 / total_pred[x]:.3f}",
                file=f,
                flush=True
            )
    torch.save(os.path.join(model_result_folder, "EfficientNet.pt"), model)


def train(
    model: nn.Module, optimizer: Any, scheduler: StepLR, device: Any,
    train_loader: DataLoader, test_loader: DataLoader, classes: list,
    loss_fn: Any, epochs: int
) -> None:

    for _ in range(epochs):
        model.train()
        loss_history = []
        for idx, data in enumerate(train_loader):

            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            loss_history.append(loss.item())

            if (idx + 1) % 100 == 0:
                print(
                    f"Train loss is {np.array(loss_history).mean():.3f}",
                    flush=True
                )
        scheduler.step()
        validate(model, test_loader, classes, device, "", True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", default="config.json", help="Path to config"
    )
    args = parser.parse_args()
    config = get_config(args.config_path)

    setup_seed(config["seed"])
    train_loader, test_loader, classes = get_dataloaders(
        config["dataset"], config["batch_size"], config["gamma"], config["phi"]
    )
    device = torch.device(
        config["device"] if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device is {device}", flush=True)

    model = EfficientNet(
        len(classes), config["alpha"], config["beta"], config["phi"]
    )
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = StepLR(optimizer, step_size=2, gamma=0.9)
    validate(model, test_loader, classes, device, "", True)
    train(
        model, optimizer, scheduler, device, train_loader, test_loader,
        classes, loss_fn, config["epochs"]
    )

    validate(
        model, test_loader, classes, device, config["result_file"],
        config["model_result_folder"]
    )


if __name__ == "__main__":
    main()
