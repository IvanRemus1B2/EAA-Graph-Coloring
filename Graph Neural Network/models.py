import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import GraphConv
from torch_geometric.nn import BatchNorm
from enum import Enum

__all__ = ["GNNRegression1", "GNNRegression2", "GNNBasicLayers"]


class ModelArchitecture(Enum):
    BasicLayers = 0


class GNNRegression1(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNNRegression1, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(1, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GNNRegression2(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNNRegression2, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(1, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GNNBasicLayers(torch.nn.Module):
    def __init__(self, device: torch.device,
                 no_units_per_gc_layer: list[int],
                 no_units_per_dense_layer: list[int],
                 layer_aggregation: str, global_layer_aggregation: str):
        super(GNNBasicLayers, self).__init__()

        self.device = device

        self.gc_layers = torch.nn.ModuleList()
        self.norm_gc_layers = torch.nn.ModuleList()
        for index in range(len(no_units_per_gc_layer) - 1):
            no_units_layer1 = no_units_per_gc_layer[index]
            no_units_layer2 = no_units_per_gc_layer[index + 1]

            self.gc_layers.append(GraphConv(no_units_layer1, no_units_layer2, aggr=layer_aggregation))
            self.norm_gc_layers.append(BatchNorm(no_units_layer2))

        self.dense_layers = torch.nn.ModuleList()
        self.norm_dense_layers = torch.nn.ModuleList()
        for index in range(len(no_units_per_dense_layer)):
            no_units_layer1 = no_units_per_dense_layer[index - 1] if index > 0 else no_units_per_gc_layer[-1]
            no_units_layer2 = no_units_per_dense_layer[index]

            self.dense_layers.append(Linear(no_units_layer1, no_units_layer2))
            self.norm_dense_layers.append(BatchNorm(no_units_layer2))

        last_layer_no_units = (
            no_units_per_dense_layer[-1] if len(no_units_per_dense_layer) > 0 else no_units_per_gc_layer[-1])
        self.regression_layer = Linear(last_layer_no_units, 1)

        self.no_units_per_gc_layer = no_units_per_gc_layer
        self.no_units_per_linear_layer = no_units_per_dense_layer

        self.layer_aggregation = layer_aggregation
        self.global_layer_aggregation = global_layer_aggregation

        self.to(device)

    def forward(self, x, edge_index, batch):
        for index in range(len(self.gc_layers)):
            x = self.gc_layers[index](x, edge_index)
            x = self.norm_gc_layers[index](x)
            x = x.relu()

        if self.global_layer_aggregation == "add":
            x = global_add_pool(x, batch)
        elif self.global_layer_aggregation == "max":
            x = global_max_pool(x, batch)
        else:
            # Assume mean
            x = global_mean_pool(x, batch)

        for index in range(len(self.dense_layers)):
            x = self.dense_layers[index](x)
            x = self.norm_dense_layers[index](x)
            x = x.relu()

        x = self.regression_layer(x)

        return x

# class GNNRegression3(torch.nn.Module):
#     def __init__(self, device: torch.device,
#                  init_no_node_features:int,
#                  no_hidden_units: list[int], layer_aggregation: str,
#                  global_layer_aggregation: str,
#                  linear_layer_dropout: float, conv_layer_dropout: float):
#         super(GNNRegression3, self).__init__()
#
#         self.device = device
#
#         self.conv1 = GraphConv(1, no_hidden_units, aggr=layer_aggregation)
#         self.batch_norm1 = BatchNorm(no_hidden_units)
#
#         self.conv2 = GraphConv(no_hidden_units, no_hidden_units, aggr=layer_aggregation)
#         self.batch_norm2 = BatchNorm(no_hidden_units)
#
#         self.conv3 = GraphConv(no_hidden_units, no_hidden_units, aggr=layer_aggregation)
#         self.batch_norm3 = BatchNorm(no_hidden_units)
#
#         self.regression_layer = Linear(no_hidden_units, 1)
#
#         self.global_layer_aggregation = global_layer_aggregation
#         self.linear_layer_dropout = linear_layer_dropout
#         self.conv_layer_dropout = conv_layer_dropout
#
#         self.to(device)
#
#     def forward(self, x, edge_index, batch):
#         x = self.conv1(x, edge_index)
#         x = self.batch_norm1(x)
#         x = F.dropout(x, p=self.conv_layer_dropout, training=self.training)
#         x = x.relu()
#
#         x = self.conv2(x, edge_index)
#         x = self.batch_norm2(x)
#         x = F.dropout(x, p=self.conv_layer_dropout, training=self.training)
#         x = x.relu()
#
#         x = self.conv3(x, edge_index)
#         x = self.batch_norm3(x)
#         x = F.dropout(x, p=self.conv_layer_dropout, training=self.training)
#
#         if self.global_layer_aggregation == "add":
#             x = global_add_pool(x, batch)
#         elif self.global_layer_aggregation == "max":
#             x = global_max_pool(x, batch)
#         else:
#             # Assume mean
#             x = global_mean_pool(x, batch)
#
#         x = F.dropout(x, p=self.linear_layer_dropout, training=self.training)
#         x = self.regression_layer(x)
#
#         return x
