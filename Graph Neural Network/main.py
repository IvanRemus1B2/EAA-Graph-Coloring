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


# instances taken from https://mat.tepper.cmu.edu/COLOR/instances.html

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
            GraphColoringInstance("", graph, chromatic_number, description="Randomly Generated", coloring=coloring))

        # random_instances.append(convert_to_data(graph, chromatic_number, no_node_features))

    return random_instances
    # execution_time = time.time() - start_time

    # print(f"Execution time:{execution_time}")


def save_instances(instances: list[GraphColoringInstance], folder: str, dataset_name: str):
    file_path = folder + "\\" + dataset_name + ".pkl"
    with open(file_path, 'wb') as file:
        pickle.dump(instances, file)
    file.close()


def load_instances(folder: str, dataset_name: str):
    file_path = folder + "\\" + dataset_name + ".pkl"
    with open(file_path, 'rb') as file:
        instances = pickle.load(file)  # deserialize using load()
    file.close()

    return instances


def split_instances(data_list, split_percent: float):
    random.shuffle(data_list)
    no_instances = len(data_list)
    split_point = int(no_instances * split_percent)

    return data_list[:split_point], data_list[split_point:]


def train(model, criterion, optimizer, train_loader):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        prediction = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        target = data.y.unsqueeze(1)
        loss = criterion(prediction, target)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(model, criterion, loader):
    model.eval()

    total_loss = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        prediction = model(data.x, data.edge_index, data.batch)
        target = data.y.unsqueeze(1)
        loss = criterion(prediction, target)
        total_loss += loss.item()
    return total_loss


def test_model(no_epochs: int,
               model, criterion, optimizer):
    random_instances = generate_random_instances(no_instances=1000,
                                                 no_nodes_interval=(30, 40),
                                                 edges_percent=(0.075, 0.3))
    train_dataset, left_instances = split_instances(random_instances, 0.7)
    val_dataset, test_dataset = split_instances(left_instances, 0.5)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for epoch in range(1, no_epochs + 1):
        train(model, criterion, optimizer, train_loader)
        train_loss = test(model, criterion, train_loader)
        val_loss = test(model, criterion, val_loader)
        test_loss = test(model, criterion, test_loader)
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}')


def inference_on(model, criterion,
                 instance_folder: str, instance_names: list[str], extension: str):
    model.eval()

    instances = read_instances(instance_names, instance_folder, extension)

    total_loss = 0
    for instance in instances:  # Iterate in batches over the training/test dataset.
        data = instance.convert_to_data()
        prediction = model(data.x, data.edge_index, torch.zeros(data.num_nodes, dtype=torch.int64))
        target = data.y.unsqueeze(0)
        loss = criterion(prediction, target)
        total_loss += loss.item()

        file_name = instance.file_name

        print(f"For {file_name}: Target:{target.item()} , Prediction:{prediction.item():.4f}")

    print(f"Total loss for files:{total_loss:.4f}")


def create_dataset():
    no_instances = 100_000
    no_nodes_interval = (30, 65)
    edges_percent = (0.075, 0.35)

    dataset_folder = "Datasets"
    dataset_name = "RG1"

    start_time = time.time()
    instances = generate_random_instances(no_instances, no_nodes_interval, edges_percent)
    print(f"Execution time {time.time() - start_time}")

    save_instances(instances, dataset_folder, dataset_name)

    # instances2 = load_instances(dataset_folder, dataset_name)
    # print(instances2)


if __name__ == '__main__':
    # graph = nx.Graph()
    #
    # graph.add_nodes_from(range(1, 5 + 1))
    #
    # graph.add_edge(1, 2)
    # graph.add_edge(2, 3)
    # graph.add_edge(3, 4)
    # graph.add_edge(4, 5)
    # graph.add_edge(5, 1)
    # # is_colorable(graph, 2, True)
    #
    # no_colors, coloring = find_chromatic_number(graph)
    # print(no_colors)
    # print(coloring)
    #
    # graph = nx.Graph()
    #
    # graph.add_nodes_from(range(1, 4 + 1))
    #
    # graph.add_edge(1, 2)
    # graph.add_edge(2, 3)
    # graph.add_edge(3, 4)
    # graph.add_edge(4, 1)
    # # is_colorable(graph, 3, False)
    # #
    # no_colors, coloring = find_chromatic_number(graph)
    # print(no_colors)
    # print(coloring)

    # instance_folder = "Instances"
    # instances_names = []
    # # instances_names += ["anna", "david", "huck", "jean","homer"]
    # # instances_names += ["zeroin.i.1", "zeroin.i.2", "zeroin.i.3"]
    # # instances_names += ["games120", "miles250"]
    # # instances_names += ["queen5_5", "queen6_6", "queen7_7", "queen8_12"]
    # # instances_names += ["myciel3", "myciel4"]
    # extension = ".col"
    # instances = read_instances(instances_names, instance_folder, extension)
    # for instance in instances:
    #     print(f"\n\nInstance name:{instance.file_name}")
    #
    #     start_time = time.time()
    #     chromatic_number, _ = find_chromatic_number(instance.graph, False)
    #     execution_time = time.time() - start_time
    #
    #     print(f"Execution time:{execution_time}")
    #     print(f"Found chromatic number:{chromatic_number}")
    #     print(f"Real chromatic number:{instance.chromatic_number}")

    instance_folder = "Instances"
    instances_names = []
    instances_names += ["anna", "david", "huck", "jean", "homer"]
    instances_names += ["zeroin.i.1", "zeroin.i.2", "zeroin.i.3"]
    instances_names += ["games120", "miles250"]
    instances_names += ["queen5_5", "queen6_6", "queen7_7", "queen8_12", "queen8_8", "queen9_9"]
    instances_names += ["myciel3", "myciel4", "myciel5", "myciel6"]
    extension = ".col"

    # no_epochs = 50
    #
    # model = GNNRegression2(hidden_channels=16)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # criterion = torch.nn.MSELoss()

    # random_instances = generate_random_instances(no_instances=500,
    #                                              no_nodes_interval=(10, 50),
    #                                              edges_percent=(0.05, 0.15))
    # train_dataset, left_instances = split_instances(random_instances, 0.95)
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # test_model(no_epochs, model, criterion, optimizer)
    #
    # inference_on(model, criterion, instance_folder, instances_names, extension)

    # instances = read_instances(instances_names, instance_folder, extension)
    # for instance in instances:
    #     no_nodes = instance.graph.number_of_nodes()
    #     no_edges = instance.graph.number_of_edges()
    #     print(f"{instance.file_name} - {no_edges / (no_nodes * (no_nodes - 1) // 2):.4f}")

    create_dataset()
