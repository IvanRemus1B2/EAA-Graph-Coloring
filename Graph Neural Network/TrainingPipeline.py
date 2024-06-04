import torch

from torch_geometric.loader import DataLoader

from models import *
from GraphColoring import *

import pickle

import wandb

import gc


def load_instances(folder: str, dataset_name: str):
    file_path = folder + "\\" + dataset_name + ".pkl"
    with open(file_path, 'rb') as file:
        instances = pickle.load(file)  # deserialize using load()
    file.close()

    return instances


def get_balanced_datasets(dataset: list[GraphColoringInstance], train_percent: float):
    no_instances = len(dataset)
    no_instances_train = int(no_instances * train_percent)
    no_instances_val = int(no_instances - no_instances_train)

    train_dataset = []
    val_dataset = []

    indexes_per_color = dict()
    for index, instance in enumerate(dataset):
        if instance.chromatic_number in indexes_per_color:
            indexes_per_color[instance.chromatic_number].append(index)
        else:
            indexes_per_color[instance.chromatic_number] = [index]

    max_indexes_per_color = 0
    for indexes in indexes_per_color.values():
        max_indexes_per_color = max(max_indexes_per_color, len(indexes))

    for index in range(max_indexes_per_color):
        for color in indexes_per_color.keys():
            if index < len(indexes_per_color[color]):
                if no_instances_train > 0:
                    no_instances_train -= 1
                    train_dataset.append(dataset[indexes_per_color[color][index]])
                elif no_instances_val > 0:
                    no_instances_train -= 1
                    val_dataset.append(dataset[indexes_per_color[color][index]])

    return train_dataset, val_dataset


class TrainingPipeline:
    def __init__(self, device: torch.device,
                 dataset_folder: str, dataset_name: str):
        dataset_instances = load_instances(dataset_folder, dataset_name)

        train_dataset, val_dataset = get_balanced_datasets(dataset_instances, 0.9)

        self.train_dataset = [instance.convert_to_data() for instance in train_dataset]
        self.val_dataset = [instance.convert_to_data() for instance in val_dataset]

        self.device = device

    def train(self, model, criterion, optimizer, train_loader):
        model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            data = data.to(model.device, non_blocking=True)
            target = data.y.unsqueeze(1)

            prediction = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.

            loss = criterion(prediction, target)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad(set_to_none=True)  # Clear gradients.

    def test(self, model, criterion, loader):
        model.eval()

        total_loss = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(model.device, non_blocking=True)
            target = data.y.unsqueeze(1)

            with torch.no_grad():
                prediction = model(data.x, data.edge_index, data.batch)

            loss = criterion(prediction, target)
            total_loss += loss.item()
        return total_loss

    def run_config(self, config=None):
        with wandb.init(config=config):
            config = wandb.config

            no_epochs = config.no_epochs
            train_batch_size = config.train_batch_size

            model = GNNBasicLayers(self.device, **config.model,
                                   layer_aggregation=config.layer_aggregation,
                                   global_layer_aggregation=config.global_layer_aggregation)

            no_workers = 2
            pin_memory = (model.device.type == 'cuda')
            persistent_workers = (no_workers != 0)
            train_loader = DataLoader(self.train_dataset, batch_size=train_batch_size, shuffle=True,
                                      pin_memory=pin_memory, num_workers=no_workers,
                                      persistent_workers=persistent_workers,
                                      drop_last=True)
            val_loader = DataLoader(self.val_dataset, batch_size=256, shuffle=False)

            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            criterion = torch.nn.MSELoss()

            for epoch in range(1, no_epochs + 1):
                self.train(model, criterion, optimizer, train_loader)
                train_loss = self.test(model, criterion, train_loader)
                val_loss = self.test(model, criterion, val_loader)
                # print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

                wandb.log({"train_loss": train_loss, "epoch": epoch})
                wandb.log({"validation_loss": val_loss, "epoch": epoch})

        torch.cuda.empty_cache()
        gc.collect()
