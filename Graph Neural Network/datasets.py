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

import matplotlib.pyplot as plt


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
            edges = list(nx.non_edges(leftover_instance.graph))
            random.shuffle(edges)
            index_edge = 0
            no_edges = len(edges)
            while index_edge < no_edges:
                left_to_add = min(no_random_edges, no_edges - index_edge)
                while left_to_add > 0:
                    u, v = edges[index_edge]
                    leftover_instance.graph.add_edge(u, v)
                    left_to_add -= 1
                    index_edge += 1

                leftover_instance.chromatic_number, leftover_instance.coloring = find_chromatic_number(
                    leftover_instance.graph)

                if leftover_instance.chromatic_number >= color_target:
                    break
        else:
            # Only add the clique,but we still need to update the chromatic number and coloring
            leftover_instance.chromatic_number, leftover_instance.coloring = find_chromatic_number(
                leftover_instance.graph)

        assigned_instances_per_color[color_target] += 1
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
    balanced_dataset_name = str(balance_type) + " " + from_dataset_named

    dataset_instances = load_instances(dataset_folder, from_dataset_named)
    print_dataset_distribution(dataset_instances, from_dataset_named)

    balanced_dataset = balance_dataset(dataset_instances, color_range, balance_type, no_random_edges, verbose)

    print_dataset_distribution(dataset_instances, balanced_dataset_name)
    save_instances(balanced_dataset, dataset_folder, balanced_dataset_name)


def generate_random_graph_v2(no_nodes: int, no_edges: int):
    graph = nx.Graph()
    nodes = range(1, no_nodes + 1)
    graph.add_nodes_from(nodes)

    possible_edges = list(nx.non_edges(graph))
    random.shuffle(possible_edges)

    for index in range(min(no_edges, len(possible_edges))):
        u, v = possible_edges[index]
        graph.add_edge(u, v)

    # Draw graph
    # nx.draw(graph, with_labels=True)
    # plt.show()

    return graph


def create_dataset_with_least_chromatic_number(no_instances: int,
                                               nodes_range: tuple[int, int], no_edges_percent: tuple[float, float],
                                               least_chromatic_number: int,
                                               balance_type: BalanceType, random_edges_to_add: int,
                                               verbose: bool = False) -> \
        list[GraphColoringInstance]:
    instances = []
    min_nodes, max_nodes = nodes_range
    min_edges_percent, max_edges_percent = no_edges_percent

    balance_str = str(balance_type).split(".")[1]
    description = f"RG C-{no_instances} N {min_nodes}-{max_nodes} E {min_edges_percent:.2f}-{max_edges_percent:.2f} {balance_str} LCN {least_chromatic_number}"

    for index in range(no_instances):
        if verbose:
            print(f"At {index}")
        no_nodes = random.randint(min_nodes, max_nodes)
        max_no_edges = no_nodes * (no_nodes - 1) / 2
        no_edges = random.randint(int(max_no_edges * min_edges_percent), int(max_no_edges * max_edges_percent))
        graph = generate_random_graph_v2(no_nodes, no_edges)

        chromatic_number, coloring = find_chromatic_number(graph, False)

        edges_to_add = list(nx.non_edges(graph))
        random.shuffle(edges_to_add)
        index_edge = 0

        while chromatic_number < least_chromatic_number:
            if balance_type == BalanceType.CLIQUE or balance_type == BalanceType.CLIQUE_RANDOM_EDGES:
                selected_nodes = random.sample(population=range(1, no_nodes + 1), k=least_chromatic_number)
                for node1 in selected_nodes:
                    for node2 in selected_nodes:
                        if node1 != node2:
                            graph.add_edge(node1, node2)

            if balance_type == BalanceType.CLIQUE_RANDOM_EDGES or balance_type == BalanceType.RANDOM_EDGES:
                left_to_add = random_edges_to_add
                while index_edge < len(edges_to_add) and left_to_add > 0:
                    u, v = edges_to_add[index_edge]
                    graph.add_edge(u, v)
                    index_edge += 1
                    left_to_add -= 1

            chromatic_number, coloring = find_chromatic_number(graph, False)

        # nx.draw(graph, with_labels=True)
        # plt.show()
        # print(index+1," ",coloring)
        # print(coloring)

        instances.append(GraphColoringInstance("", graph, chromatic_number, description, "", coloring))

    return instances


if __name__ == '__main__':
    start_time = time.time()

    dataset_folder = "Datasets"
    # dataset_name = "RG1 10k N 30-60 E 7,5-20"
    #
    # color_range = [3, 8]
    # balance_type = BalanceType.RANDOM_EDGES
    # no_random_edges = 15
    #
    # create_balanced_dataset_from(dataset_name, color_range, balance_type, no_random_edges, True)

    #

    least_chromatic_number = 5
    no_instances = 10_000
    instances = create_dataset_with_least_chromatic_number(no_instances, (30, 80), (0.05, 0.2), least_chromatic_number,
                                                           BalanceType.RANDOM_EDGES, 25, True)

    save_instances(instances, dataset_folder, f"RG1 C-{no_instances} LCN-{least_chromatic_number}")

    # instances = load_instances(dataset_folder, "RG1 1 LCN-5")

    print(instances)
    end_time = time.time()
    print("Execution time: ", end_time - start_time)

    # instances = read_instances(instances_names, instance_folder, extension)
    # for instance in instances:
    #     no_nodes = instance.graph.number_of_nodes()
    #     no_edges = instance.graph.number_of_edges()
    #     print(f"{instance.file_name} - {no_edges / (no_nodes * (no_nodes - 1) // 2):.4f}")

    # create_dataset()

    # graph = nx.Graph()
    #
    # graph.add_nodes_from(range(1, 60))

    # graph.add_edge(1, 2)
    # graph.add_edge(2, 3)
    # graph.add_edge(3, 4)
    # graph.add_edge(4, 5)
    # graph.add_edge(5, 1)

    # print(len(list(nx.non_edges(graph))))
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
    #
    # print(graph.edges())
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
