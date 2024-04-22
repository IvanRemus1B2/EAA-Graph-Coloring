from typing import Union
import networkx as nx
import torch
from torch_geometric.data import Data

__all__ = ["GraphColoringInstance"]


class GraphColoringInstance:
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
