from ortools.sat.python import cp_model
import time

import networkx as nx

from typing import Union


# instances taken from https://mat.tepper.cmu.edu/COLOR/instances.html

class GraphColoringInstance:
    def __init__(self, file_name: str, graph: nx.Graph, chromatic_number: Union[int, None], description: str,
                 source: str):
        self.file_name = file_name
        self.graph = graph
        self.chromatic_number = chromatic_number
        if chromatic_number is None:
            self.chromatic_number = "Unknown"

        self.description = description
        self.source = source

    def __str__(self):
        representation = ""
        representation += f"\n\nFor instance {self.file_name} from {self.source}"
        representation += f"\nDescription:\n{self.description}"
        representation += f"\nGraph:{self.graph}"
        representation += f"\nChromatic Number:{self.chromatic_number}"
        return representation

    def __repr__(self):
        return self.__str__()


# Used for extracting the solution
class MapColoringSolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, graph: nx.Graph, color_assignments: list, show: bool = False):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__graph = graph
        self.__color_assignments = color_assignments

        self.__show = show

        self.__no_solutions = 1

        self.__start_time = time.time()
        self.__search_time = time.time()

        self.solution = [-1 for _ in range(graph.number_of_nodes() + 1)]

    @property
    def solution_count(self) -> int:
        return self.__no_solutions

    def on_solution_callback(self):
        if self.__show:
            self.__search_time = time.time() - self.__start_time
            print(f"\nSolution {self.__no_solutions}")

        if self.__no_solutions == 1:
            for index, node in enumerate(self.__color_assignments, 1):
                if self.__show:
                    print(f"{node}={self.Value(node)} , ", end="")
                self.solution[index] = self.Value(node)

        self.__no_solutions += 1

        if self.__show:
            print()

        self.__show = False
        self.StopSearch()


def is_colorable(graph: nx.Graph, no_colors: int, verbose: bool = False) -> tuple[bool, Union[None, list[int]]]:
    """
    Check whether a graph is colorable with no_colors
    :param graph: the graph
    :param no_colors: the number of colors to test
    :param verbose: whether we show the description of the instance and results on the console
    :return: a tuple with the info of whether we found a coloring.If True,we also return the coloring.
    The coloring will be a list of no_vertices+1 with meaning of c_i=color of node i from 0 to no_nodes
    Node 0 is by default -1.Colors start from 0 to no_colours-1
    """
    if verbose:
        print("\nNo nodes:", graph.number_of_nodes())
        print("No edges:", graph.number_of_edges())
        print("Edges:", graph.edges)
        print("No of colors:", no_colors)

    model = cp_model.CpModel()
    no_nodes = graph.number_of_nodes()

    color_assignments = [model.NewIntVar(0, no_colors - 1, "Node_" + str(index + 1)) for index in range(no_nodes)]

    for (node1, node2) in graph.edges:
        model.Add(color_assignments[node1 - 1] != color_assignments[node2 - 1])

    solver = cp_model.CpSolver()
    solution_printer = MapColoringSolutionPrinter(graph, color_assignments, verbose)

    status = solver.Solve(model, solution_printer)
    if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
        if verbose:
            print(f"No {no_colors}-coloring found")
        return False, None

    coloring = solution_printer.solution
    if verbose:
        print(f"Found a {no_colors}-coloring")
        print(f"{no_colors}-coloring:", coloring)

    return True, coloring


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


def find_chromatic_number(graph: nx.Graph, verbose: bool = False) -> tuple[int, list[int]]:
    """
    Find the minimum number c with s.t. there exists a c-coloring of the graph.This solution uses binary
    search ( ͡° ͜ʖ ͡°)
    :param graph: the graph
    :return: a tuple with the minimum color number and the coloring(as proof)
    """
    start_time = time.time()

    no_nodes = graph.number_of_nodes()
    left = 1
    right = no_nodes
    valid_coloring = [index - 1 for index in range(no_nodes + 1)]
    while left < right:
        possible_color = (left + right) // 2
        exists_color_assignments, coloring = is_colorable(graph, possible_color, verbose)
        if exists_color_assignments:
            right = possible_color
            valid_coloring = coloring
        else:
            left = possible_color + 1

    if verbose:
        print(f"Execution time:{time.time() - start_time}")

    return right, valid_coloring


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
    return instances


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

    instance_folder = "Instances"
    instances_names = ["queen8_8"]
    extension = ".col"
    instances = read_instances(instances_names, instance_folder, extension)
    for instance in instances:
        print(f"\n\nInstance name:{instance.file_name}")
        chromatic_number, _ = find_chromatic_number(instance.graph, True)
        print(f"\nFound chromatic number:{chromatic_number}")
        print(f"\nReal chromatic number:{instance.chromatic_number}")
