from ortools.sat.python import cp_model
import time

import networkx as nx

from typing import Union

import random

import torch
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from GCConstraintSatisfaction import find_chromatic_number
from models import *
from GraphColoring import *

import pickle

import wandb

from TrainingPipeline import TrainingPipeline

import numpy as np

from enum import Enum

from models import ModelArchitecture


# instances taken from https://mat.tepper.cmu.edu/COLOR/instances.html

class GraphColoringInstanceV2:
    def __init__(self, file_name: str,
                 graph: nx.Graph, chromatic_number: Union[int, None],
                 description: Union[str, None] = None, source: Union[str, None] = None,
                 coloring: Union[list[int], None] = None):
        self.file_name = file_name
        self.graph = graph
        self.chromatic_number = chromatic_number
        if chromatic_number is None:
            self.chromatic_number = "Unknown"
        self.coloring = coloring

        self.description = description
        self.source = source

    def convert_to_data(self, no_node_features: int = 1) -> Data:
        # Create tensors for graph connectivity
        edge_index = []
        for edge in self.graph.edges():
            node1, node2 = edge[0] - 1, edge[1] - 1
            edge_index.append([node1, node2])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Create labels as the chromatic number
        labels = torch.tensor(self.chromatic_number, dtype=torch.float)

        features = torch.ones((self.graph.number_of_nodes(), no_node_features), dtype=torch.float)

        return Data(x=features, edge_index=edge_index, y=labels, instance_file=self.file_name)

    def __str__(self):
        representation = ""
        representation += f"\n\nFor instance {self.file_name} from {self.source}"
        representation += f"\nDescription:\n{self.description}"
        representation += f"\nGraph:{self.graph}"
        representation += f"\nChromatic Number:{self.chromatic_number}"
        return representation

    def __repr__(self):
        return self.__str__()


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
        # For multi-gpu workstations, PyTorch will use the first available GPU (cuda:0), unless specified otherwise
        # (cuda:1).
    if torch.backends.mps.is_available():
        return torch.device('mos')
    return torch.device('cpu')


def create_graph(no_vertices: int, edges: list[tuple[int, int]]) -> nx.Graph:
    """
    Create a graph as a nx.Graph instance.We assume nodes are from indexed from 1 to no_vertices
    :param no_vertices: the number of vertices/nodes
    :param edges: the number of edges
    :return: a nx.Graph instance
    """
    graph = nx.Graph()

    graph.add_nodes_from(range(1, no_vertices + 1))

    graph.add_edges_from(edges)

    return graph


def read_col_file(folder: str, instance_name: str, extension: str) -> GraphColoringInstance:
    # Search for the chromatic number and source in the info file
    instance_chromatic_number = None
    found = False

    info_file_path = folder + "\\" + "Instances Info.txt"
    file_path = folder + "\\" + instance_name + extension
    with open(info_file_path, 'r') as info_file:
        for line in info_file.readlines():
            name, _, chromatic_number, source = line.split(",")
            name = name.rsplit(".", 1)[0]
            chromatic_number = int(chromatic_number) if chromatic_number != " ?" else None
            if name == instance_name:
                instance_chromatic_number = chromatic_number
                found = True
                break

    if not found:
        raise ValueError(f"The file {instance_name} wasn't found in the Instance Info.txt file")

    with open(file_path, 'r') as file:
        lines = file.readlines()
        no_vertices = None
        no_edges = None
        description = ""
        edges = []
        for line in lines:
            if line.startswith('p'):
                no_vertices, no_edges = map(int, line.split()[2:])
            elif line.startswith('e'):
                node1, node2 = map(int, line.split()[1:])
                edges.append((node1, node2))
            elif line.startswith('c'):
                description += line[2:]
            else:
                raise ValueError(f"The file {instance_name} has a line that doesn't start with 'c','p' or 'e'")

        if no_vertices is None or no_edges is None:
            raise ValueError(f"No line that starts with 'p' was found for the {instance_name}")

    return GraphColoringInstance(instance_name, create_graph(no_vertices, edges), instance_chromatic_number,
                                 description, source)


def read_instances(instance_names: list[str], instance_folder: str, extension: str) -> list[GraphColoringInstance]:
    instances = []
    for instance_name in instance_names:
        coloring_instance = read_col_file(instance_folder, instance_name, extension)
        instances.append(coloring_instance)
        # instances.append(coloring_instance.convert_to_data())
    return instances


def generate_random_graph(n_nodes, n_edges):
    G = nx.Graph()
    nodes = range(1, n_nodes + 1)
    G.add_nodes_from(nodes)

    edges = set()
    while len(edges) < n_edges:
        edge = (random.randint(1, n_nodes), random.randint(1, n_nodes))
        if edge[0] != edge[1]:
            edges.add(edge)
    G.add_edges_from(edges)

    # Draw graph
    # nx.draw(G, with_labels=True)
    # plt.show()

    return G


def convert_to_data(graph: nx.Graph, chromatic_number: int, no_node_features: int = 1):
    # Create tensors for graph connectivity
    edge_index = []
    for edge in graph.edges():
        node1, node2 = edge[0] - 1, edge[1] - 1
        edge_index.append([node1, node2])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create labels as the chromatic number
    labels = torch.tensor(chromatic_number, dtype=torch.float)

    features = torch.ones((graph.number_of_nodes(), no_node_features), dtype=torch.float)

    return Data(x=features, edge_index=edge_index, y=labels)


def generate_random_instances(no_instances: int,
                              no_nodes_interval: tuple[int, int], edges_percent: tuple[float, float],
                              no_node_features: int = 1):
    random_instances = []
    start_time = time.time()

    min_nodes, max_nodes = no_nodes_interval
    min_edges_percent, max_edges_percent = edges_percent

    for i in range(no_instances):
        num_nodes = random.randint(min_nodes, max_nodes)
        max_num_edges = num_nodes * (num_nodes - 1) / 2
        num_edges = random.randint(int(max_num_edges * min_edges_percent), int(max_num_edges * max_edges_percent))
        graph = generate_random_graph(num_nodes, num_edges)

        chromatic_number, coloring = find_chromatic_number(graph, False)

        random_instances.append(
            GraphColoringInstance("", graph, chromatic_number, description="RG", coloring=coloring))

        print(f"At {i}")

        # random_instances.append(convert_to_data(graph, chromatic_number, no_node_features))

    return random_instances
    # execution_time = time.time() - start_time

    # print(f"Execution time:{execution_time}")


def save_instances(instances: list[GraphColoringInstance], folder: str, dataset_name: str):
    file_path = folder + "\\" + dataset_name + ".pkl"
    with open(file_path, 'wb') as file:
        pickle.dump(instances, file)
    file.close()


def save_instances_as_V2(instances: list[GraphColoringInstance], folder: str, dataset_name: str):
    file_path = folder + "\\" + dataset_name + ".pkl"
    instances = [
        GraphColoringInstanceV2(instance.file_name, instance.graph, instance.chromatic_number, instance.description,
                                instance.source, instance.coloring) for instance in instances]
    with open(file_path, 'wb') as file:
        pickle.dump(instances, file)
    file.close()


def load_instances(folder: str, dataset_name: str):
    file_path = folder + "\\" + dataset_name + ".pkl"
    with open(file_path, 'rb') as file:
        instances = pickle.load(file)  # deserialize using load()
    file.close()

    return instances


def split_instances(data_list, train_percent: float):
    label_positions = dict()
    all_labels = []
    no_instances = len(data_list)
    for index in range(no_instances):
        label = int(data_list[index].y.item())
        if label in label_positions:
            label_positions[label].append(index)
        else:
            label_positions[label] = [index]
            all_labels.append(label)

    for label in all_labels:
        random.shuffle(label_positions[label])

    train_instances = []
    val_instances = []

    found = True
    position = 0
    add_train_left = int(train_percent * no_instances)
    while found:
        found = False
        for _, positions in label_positions.items():
            if position < len(positions):
                found = True
                if add_train_left > 0:
                    train_instances.append(data_list[positions[position]])
                    add_train_left -= 1
                else:
                    val_instances.append(data_list[positions[position]])
        position += 1

    return train_instances, val_instances


def train(model, criterion, optimizer, train_loader, ):
    model.train()

    total_loss = 0.0
    no_batches = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(model.device, non_blocking=True)
        target = data.y.unsqueeze(1)

        prediction = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.

        loss = criterion(prediction, target)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        total_loss += loss.item()
        optimizer.zero_grad(set_to_none=True)  # Clear gradients.

    return total_loss


def test(model, criterion, loader):
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


def test_model(no_epochs: int, train_batch_size: int,
               instances: Union[list[Data], None],
               train_percent: float,
               model, criterion, optimizer):
    train_dataset, val_dataset = split_instances(instances, train_percent)
    no_train_instances, no_val_instances = len(train_dataset), len(val_dataset)

    no_workers = 1
    pin_memory = (model.device.type == 'cuda')
    persistent_workers = (no_workers != 0)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                              pin_memory=pin_memory, num_workers=no_workers, persistent_workers=persistent_workers,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    for epoch in range(1, no_epochs + 1):
        train_loss = train(model, criterion, optimizer, train_loader) / no_train_instances
        val_loss = test(model, criterion, val_loader) / no_val_instances
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


def inference_on(model, criterion,
                 instance_folder: str, instance_names: list[str], extension: str):
    print("\nInference:")
    model.eval()

    instances = read_instances(instance_names, instance_folder, extension)

    init_no_node_features = model.no_units_per_gc_layer[0]

    errors = np.zeros(len(instance_names))

    total_loss = 0
    for index, instance in enumerate(instances):  # Iterate in batches over the training/test dataset.
        data = instance.convert_to_data(no_node_features=init_no_node_features).to(model.device, non_blocking=True)

        with torch.no_grad():
            prediction = model(data.x, data.edge_index,
                               torch.zeros(data.num_nodes, dtype=torch.int64).to(model.device,
                                                                                 non_blocking=True)).squeeze(0)

        target = data.y.unsqueeze(0)

        loss = criterion(prediction, target)
        total_loss += loss.item()
        errors[index] = np.abs((target - prediction).item())

        file_name = instance.file_name

        print(f"{file_name} -> Target:{target.item()} , Prediction:{prediction.item():.4f} , Error:{errors[index]:.4f}")

    print(f"Total loss for files: {total_loss:.4f}")
    print(f"Total MAE score: {np.sum(errors)}")
    print(f"All errors: {errors}")


def create_dataset():
    no_instances = 10_000
    no_nodes_interval = (30, 60)
    edges_percent = (0.075, 0.2)

    dataset_folder = "Datasets"
    dataset_name = "T1 RG 10k N 30-60 E 7,5-20"

    # start_time = time.time()
    instances = generate_random_instances(no_instances, no_nodes_interval, edges_percent)
    # print(f"Execution time {time.time() - start_time}")

    save_instances(instances, dataset_folder, dataset_name)

    # instances2 = load_instances(dataset_folder, dataset_name)
    # print(instances2)


class BalanceType(Enum):
    CLIQUE = 0,
    CLIQUE_RANDOM_EDGES = 1,
    RANDOM_EDGES = 2


def balance_dataset(dataset: list[GraphColoringInstance], color_range: list[int, int],
                    balance_type: BalanceType, no_random_edges: int,
                    verbose: bool = 0):
    random.shuffle(dataset)

    start_color, end_color = color_range
    no_colors = end_color - start_color + 1
    no_instances = len(dataset)
    instances_per_color = no_instances // no_colors
    leftover = []

    new_dataset = []
    assigned_instances_per_color = dict()
    for index in range(start_color, end_color + 1):
        assigned_instances_per_color[index] = 0

    for instance in dataset:
        instance_color = instance.chromatic_number
        if instance_color < start_color or instance_color > end_color:
            leftover.append(instance)
            continue

        count = assigned_instances_per_color.setdefault(instance_color, 0)
        if count < instances_per_color:
            new_dataset.append(instance)
            assigned_instances_per_color[instance_color] = count + 1
        else:
            leftover.append(instance)

    leftover.sort(key=lambda instance: instance.chromatic_number)

    # print(assigned_instances_per_color)
    if verbose:
        print(f"Total: {len(leftover)}")
    color_target = start_color
    for index, leftover_instance in enumerate(leftover):
        if verbose and index % 20 == 0:
            print(f"At {index}")

        while color_target <= end_color and assigned_instances_per_color[color_target] >= instances_per_color:
            color_target += 1
        color_target = min(end_color, color_target)

        no_nodes = leftover_instance.graph.number_of_nodes()
        if balance_type == BalanceType.CLIQUE or balance_type == BalanceType.CLIQUE_RANDOM_EDGES:
            selected_nodes = random.sample(population=range(1, no_nodes + 1), k=color_target)
            for node1 in selected_nodes:
                for node2 in selected_nodes:
                    if node1 != node2:
                        leftover_instance.graph.add_edge(node1, node2)

        if balance_type == BalanceType.CLIQUE_RANDOM_EDGES or balance_type == BalanceType.RANDOM_EDGES:
            # Add 40 more edges at random
            for index2 in range(no_random_edges):
                node1, node2 = random.sample(population=range(1, no_nodes + 1), k=2)
                if node1 != node2:
                    leftover_instance.graph.add_edge(node1, node2)

        assigned_instances_per_color[color_target] += 1
        leftover_instance.chromatic_number, leftover_instance.coloring = find_chromatic_number(leftover_instance.graph)

        new_dataset.append(leftover_instance)

    return new_dataset


def print_dataset_distribution(dataset: list[GraphColoringInstance], dataset_name: str):
    print(f"For {dataset_name}")
    class_count = dict()
    for instance in dataset:
        value = class_count.setdefault(instance.chromatic_number, 0)
        class_count[instance.chromatic_number] = value + 1

    print(class_count)


def create_balanced_dataset_from(from_dataset_named: str, color_range: list[int, int], balance_type: BalanceType,
                                 no_random_edges: int, verbose: bool = False):
    dataset_folder = "Datasets"

    dataset_instances = load_instances(dataset_folder, from_dataset_named)
    print_dataset_distribution(dataset_instances, from_dataset_named)
    balanced_dataset = balance_dataset(dataset_instances, color_range, balance_type, no_random_edges, verbose)

    save_instances(balanced_dataset, dataset_folder, str(balance_type) + " " + from_dataset_named)


def get_sweep_params():
    parameters_dict = {
        'no_epochs': {
            'value': 100
        },
        'model': {
            'value': {
                'no_hidden_units': 80,
                'linear_layer_dropout': 0.25,
                'conv_layer_dropout': 0.1
            }
        },
        'layer_aggregation': {
            'values': ["add", "max", "mean"]
        },
        'global_layer_aggregation': {
            'values': ["add", "max", "mean"]
        },
        'train_batch_size': {
            'values': [64, 128, 256, 512]
        },
        'lr': {
            'value': 1e-3
        },
        'weight_decay': {
            'distribution': 'uniform',
            'min': 1e-4,
            'max': 1e-1
        }
    }

    return parameters_dict


def run_sweep(dataset_folder: str, dataset_name: str,
              no_searches: int, sweep_id: str = None,
              device: torch.device = get_default_device()):
    # Step 1
    wandb.login()

    sweep_config = {
        'method': 'random'
    }

    metric = {
        'name': 'validation_loss',
        'goal': 'minimize'
    }

    sweep_config['metric'] = metric

    sweep_config['parameters'] = get_sweep_params()

    # Step 2
    project_name = "EAA Graph Coloring Fine Tuning "
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep_config, project=project_name)

    print("Project name: ", project_name)
    print("Sweep Id: ", sweep_id)

    training_pipeline = TrainingPipeline(device, dataset_folder, dataset_name)

    # Step 3
    wandb.agent(sweep_id, training_pipeline.run_config, count=no_searches)


def save_model(model, model_path: str):
    torch.save(model, model_path + ".pth")
    # model_scripted = torch.jit.script(model)  # Export to TorchScript
    # model_scripted.save(model_path + ".pt")  # Save


def load_model(model_path: str):
    # model = torch.jit.load(model_path + ".pt")
    # model.eval()
    model = torch.load(model_path + ".pth")
    model.eval()
    return model


def create_train_model():
    instance_folder = "Instances"
    dataset_folder = "Datasets"
    models_folder = "Models"

    instances_names = []
    instances_names += ["anna", "david", "huck", "jean", "homer"]
    # instances_names += ["zeroin.i.1", "zeroin.i.2", "zeroin.i.3"]
    # instances_names += ["games120", "miles250"]
    instances_names += ["queen5_5", "queen6_6", "queen7_7", "queen8_12", "queen8_8", "queen9_9", "queen13_13"]
    instances_names += ["myciel5", "myciel6", "myciel7"]
    instances_names += ["games120"]
    extension = ".col"

    start_time = time.time()
    # device = torch.device('cpu')
    device = get_default_device()

    no_epochs = 10
    train_batch_size = 64
    train_percent = 0.9

    dataset_name = "RE B 100k with 3-6 CN"
    # dataset_name = "RG1 C-100 LCN-6"
    # dataset_name = "RG2 100k N 20-60 E 7,5-20"

    model_name = "RE 3"
    model_architecture = ModelArchitecture.BasicLayers

    hyper_parameters = {
        'no_units_per_gc_layer': [128, 128, 128],
        'no_node_features': 128,

        'no_units_per_dense_layer': [],

        'layer_aggregation': "add",
        'global_layer_aggregation': "mean",
    }
    # linear_layer_dropout = 0.25
    # conv_layer_dropout = 0.1

    learning_rate = 1e-3

    # -----------------

    print(f"For device: {device}")
    print(f"Dataset name:{dataset_name}")
    model_architecture_str = str(model_architecture).split(".")[1]
    print(f"no epochs:{no_epochs},batch_size:{train_batch_size},train_percentage:{train_percent}")

    model = GNNBasicLayers(device, no_units_per_gc_layer=hyper_parameters['no_units_per_gc_layer'],
                           no_units_per_dense_layer=hyper_parameters['no_units_per_dense_layer'],
                           layer_aggregation=hyper_parameters['layer_aggregation'],
                           global_layer_aggregation=hyper_parameters['global_layer_aggregation'])

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Name:{model_name}")
    print(f"type:{model_architecture_str}")
    print(f"Hyper parameters:{hyper_parameters}")
    # print(f"linear_layer_dropout:{linear_layer_dropout},conv_layer_dropout:{conv_layer_dropout}")
    print(f"No Params:{params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    dataset = load_instances(dataset_folder, dataset_name)
    print_dataset_distribution(dataset, dataset_name)

    dataset_instances = [instance.convert_to_data(hyper_parameters['no_node_features']) for instance in dataset]
    del dataset

    test_model(no_epochs, train_batch_size,
               dataset_instances, train_percent,
               model, criterion, optimizer)

    model_path = (
                     "" if models_folder == "" else models_folder + "/") + model_architecture_str + "-" + model_name

    save_model(model, model_path)

    test_model_from(instance_folder, instances_names, models_folder, model_architecture_str, model_name, extension)

    print(f"Execution time:{time.time() - start_time:.4f}")


def test_model_from(instances_folder: str, instances_names: list[str],
                    models_folder: str, model_architecture_str: str, model_name: str,
                    extension: str = ".col"):
    start_time = time.time()

    model_path = (
                     "" if models_folder == "" else models_folder + "/") + model_architecture_str + "-" + model_name
    model = load_model(model_path)

    criterion = torch.nn.MSELoss()

    inference_on(model, criterion, instances_folder, instances_names, extension)

    print(f"Execution time:{time.time() - start_time:.4f}")


if __name__ == '__main__':
    create_train_model()
    # main()
