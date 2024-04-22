from typing import Union
from ortools.sat.python import cp_model
import networkx as nx
import time


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

    for (node1, node2) in graph.edges():
        model.Add(color_assignments[node1 - 1] != color_assignments[node2 - 1])

    solver = cp_model.CpSolver()
    solution_printer = MapColoringSolutionPrinter(graph, color_assignments, verbose)
    solver.parameters.num_search_workers = 8

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
