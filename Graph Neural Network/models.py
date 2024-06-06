import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import LSTMAggregation
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import GraphConv
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import aggr
from enum import Enum
from torch_geometric.utils import sort_edge_index


class ModelArchitecture(Enum):
    GraphConv = 0,
    SAGEConv = 1,
    SAGEConv_LSTM = 2,
    GraphConv_LSTM = 3


def get_model(model_architecture_type: ModelArchitecture,
              device: torch.device, hyper_parameters: dict) -> torch.nn.Module:
    if model_architecture_type == ModelArchitecture.GraphConv:
        return GNN_GraphConv(device, no_units_per_gc_layer=hyper_parameters['no_units_per_gc_layer'],
                             no_units_per_dense_layer=hyper_parameters['no_units_per_dense_layer'],
                             layer_aggregation=hyper_parameters['layer_aggregation'],
                             global_layer_aggregation=hyper_parameters['global_layer_aggregation'],
                             gc_layer_dropout=hyper_parameters['gc_layer_dropout'])
    # elif model_architecture_type == ModelArchitecture.AttentionLayers:
    #     return GNNAttention(device,
    #                         no_node_features=hyper_parameters['no_node_features'],
    #                         layer_aggregation=hyper_parameters['layer_aggregation'],
    #                         final_layer_aggregation=hyper_parameters['final_layer_aggregation'],
    #                         layers_dropout=hyper_parameters['layers_dropout'])
    # elif model_architecture_type == ModelArchitecture.AttentionLayersTest:
    #     return GAT(device,
    #                no_node_features=hyper_parameters['no_node_features'])
    elif model_architecture_type == ModelArchitecture.SAGEConv:
        return GNN_SAGEConv(device, no_units_per_gc_layer=hyper_parameters['no_units_per_gc_layer'],
                            no_units_per_dense_layer=hyper_parameters['no_units_per_dense_layer'],
                            layer_aggregation=hyper_parameters['layer_aggregation'],
                            global_layer_aggregation=hyper_parameters['global_layer_aggregation'],
                            gc_layer_dropout=hyper_parameters['gc_layer_dropout'])
    elif model_architecture_type == ModelArchitecture.GraphConv_LSTM:
        return GNN_GraphConv_LSTM(device, no_units_per_gc_layer=hyper_parameters['no_units_per_gc_layer'],
                                  no_units_per_dense_layer=hyper_parameters['no_units_per_dense_layer'],
                                  global_layer_aggregation=hyper_parameters['global_layer_aggregation'],
                                  gc_layer_dropout=hyper_parameters['gc_layer_dropout'])
    elif model_architecture_type == ModelArchitecture.SAGEConv_LSTM:
        return GNN_SAGEConv_LSTM(device, no_units_per_gc_layer=hyper_parameters['no_units_per_gc_layer'],
                                 no_units_per_dense_layer=hyper_parameters['no_units_per_dense_layer'],
                                 global_layer_aggregation=hyper_parameters['global_layer_aggregation'],
                                 gc_layer_dropout=hyper_parameters['gc_layer_dropout'])
    return None


class GNNSimpleLayers(torch.nn.Module):
    def __init__(self, device: torch.device,
                 no_units_per_gc_layer: list[int],
                 no_units_per_dense_layer: list[int],
                 layer_aggregation: str, global_layer_aggregation: str,
                 gc_layer_dropout: float):
        super(GNNSimpleLayers, self).__init__()

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
        self.gc_layer_dropout = gc_layer_dropout

        self.to(device)

    def forward(self, x, edge_index, batch):
        for index in range(len(self.gc_layers)):
            x = self.gc_layers[index](x, edge_index)
            x = self.norm_gc_layers[index](x)
            x = F.dropout(x, p=self.gc_layer_dropout, training=self.training)
            x = torch.nn.LeakyReLU()(x)

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
            x = torch.nn.LeakyReLU()(x)

        x = self.regression_layer(x)

        return x


class GNN_GraphConv(torch.nn.Module):
    def __init__(self, device: torch.device,
                 no_units_per_gc_layer: list[int],
                 no_units_per_dense_layer: list[int],
                 layer_aggregation: str, global_layer_aggregation: str,
                 gc_layer_dropout: float):
        super(GNN_GraphConv, self).__init__()

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
        self.gc_layer_dropout = gc_layer_dropout

        self.to(device)

    def forward(self, x, edge_index, batch):
        for index in range(len(self.gc_layers)):
            x = self.gc_layers[index](x, edge_index)
            x = self.norm_gc_layers[index](x)
            x = F.dropout(x, p=self.gc_layer_dropout, training=self.training)
            x = torch.nn.LeakyReLU()(x)

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
            x = torch.nn.LeakyReLU()(x)

        x = self.regression_layer(x)

        return x


class GNN_SAGEConv(torch.nn.Module):
    def __init__(self, device: torch.device,
                 no_units_per_gc_layer: list[int],
                 no_units_per_dense_layer: list[int],
                 layer_aggregation: str, global_layer_aggregation: str,
                 gc_layer_dropout: float):
        super(GNN_SAGEConv, self).__init__()

        self.device = device

        self.gc_layers = torch.nn.ModuleList()
        self.norm_gc_layers = torch.nn.ModuleList()
        for index in range(len(no_units_per_gc_layer) - 1):
            no_units_layer1 = no_units_per_gc_layer[index]
            no_units_layer2 = no_units_per_gc_layer[index + 1]

            self.gc_layers.append(SAGEConv(no_units_layer1, no_units_layer2, aggr=layer_aggregation))
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
        self.gc_layer_dropout = gc_layer_dropout

        self.to(device)

    def forward(self, x, edge_index, batch):
        for index in range(len(self.gc_layers)):
            x = self.gc_layers[index](x, edge_index)
            x = self.norm_gc_layers[index](x)
            x = F.dropout(x, p=self.gc_layer_dropout, training=self.training)
            x = torch.nn.LeakyReLU()(x)

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
            x = torch.nn.LeakyReLU()(x)

        x = self.regression_layer(x)

        return x


class GNN_SAGEConv_LSTM(torch.nn.Module):
    def __init__(self, device: torch.device,
                 no_units_per_gc_layer: list[int],
                 no_units_per_dense_layer: list[int],
                 global_layer_aggregation: str,
                 gc_layer_dropout: float):
        super(GNN_SAGEConv_LSTM, self).__init__()

        self.device = device

        self.gc_layers = torch.nn.ModuleList()
        self.norm_gc_layers = torch.nn.ModuleList()
        for index in range(len(no_units_per_gc_layer) - 1):
            no_units_layer1 = no_units_per_gc_layer[index]
            no_units_layer2 = no_units_per_gc_layer[index + 1]

            self.gc_layers.append(
                SAGEConv(no_units_layer1, no_units_layer2, aggr=LSTMAggregation(no_units_layer1, no_units_layer2)))
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

        self.global_layer_aggregation = global_layer_aggregation
        self.gc_layer_dropout = gc_layer_dropout

        self.to(device)

    def forward(self, x, edge_index, batch):
        sorted_edge_index = sort_edge_index(edge_index, num_nodes=len(batch), sort_by_row=False)

        for index in range(len(self.gc_layers)):
            x = self.gc_layers[index](x, sorted_edge_index)
            x = self.norm_gc_layers[index](x)
            x = F.dropout(x, p=self.gc_layer_dropout, training=self.training)
            x = torch.nn.LeakyReLU()(x)

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
            x = torch.nn.LeakyReLU()(x)

        x = self.regression_layer(x)

        return x


class GNN_GraphConv_LSTM(torch.nn.Module):
    def __init__(self, device: torch.device,
                 no_units_per_gc_layer: list[int],
                 no_units_per_dense_layer: list[int],
                 global_layer_aggregation: str,
                 gc_layer_dropout: float):
        super(GNN_GraphConv_LSTM, self).__init__()

        self.device = device

        self.gc_layers = torch.nn.ModuleList()
        self.norm_gc_layers = torch.nn.ModuleList()
        for index in range(len(no_units_per_gc_layer) - 1):
            no_units_layer1 = no_units_per_gc_layer[index]
            no_units_layer2 = no_units_per_gc_layer[index + 1]

            self.gc_layers.append(
                GraphConv(no_units_layer1, no_units_layer2, aggr=LSTMAggregation(no_units_layer1, no_units_layer2)))
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

        self.global_layer_aggregation = global_layer_aggregation
        self.gc_layer_dropout = gc_layer_dropout

        self.to(device)

    def forward(self, x, edge_index, batch):
        sorted_edge_index = sort_edge_index(edge_index, num_nodes=len(batch), sort_by_row=False)

        for index in range(len(self.gc_layers)):
            x = self.gc_layers[index](x, sorted_edge_index)
            x = self.norm_gc_layers[index](x)
            x = F.dropout(x, p=self.gc_layer_dropout, training=self.training)
            x = torch.nn.LeakyReLU()(x)

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
            x = torch.nn.LeakyReLU()(x)

        x = self.regression_layer(x)

        return x


class GNNAttention(torch.nn.Module):
    def __init__(self, device: torch.device,
                 no_node_features: int,
                 layer_aggregation: str, final_layer_aggregation: str,
                 layers_dropout: float):
        super(GNNAttention, self).__init__()

        self.device = device

        self.conv1_out = 8

        self.in_head = 8
        self.out_head = 1

        self.basic_layer = GraphConv(no_node_features, 128, aggr=layer_aggregation)
        self.conv1 = GATConv(128, self.conv1_out, heads=self.in_head, dropout=layers_dropout)
        self.conv2 = GATConv(self.conv1_out * self.in_head, 64, concat=False,
                             heads=self.out_head, dropout=layers_dropout)

        self.final_layer_aggregation = final_layer_aggregation
        self.layers_dropout = layers_dropout
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)

        self.regression_layer = Linear(64, 1)

        self.to(device)

    def forward(self, x, edge_index):
        # Dropout before the GAT layer is used to avoid overfitting
        x = F.dropout(x, p=self.layers_dropout, training=self.training)
        x = self.basic_layer(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.layers_dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.layers_dropout, training=self.training)
        x = self.conv2(x, edge_index)

        if self.final_layer_aggregation == "add":
            x = aggr.SumAggregation()(x)
        elif self.final_layer_aggregation == "max":
            x = aggr.MaxAggregation()(x)
        else:
            # Assume mean
            x = aggr.MeanAggregation()(x)

        x = self.regression_layer(x)

        return x


class GAT(torch.nn.Module):
    def __init__(self, device: torch.device, no_node_features: int):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        self.conv2_out_channels = 32

        self.conv1 = GATConv(no_node_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid * self.in_head, self.conv2_out_channels, concat=False,
                             heads=self.out_head, dropout=0.6)

        self.regression_layer = Linear(self.out_head * self.conv2_out_channels, 1)

        self.device = device
        self.to(device)

    def forward(self, x, edge_index):
        # Dropout before the GAT layer is used to avoid overfitting
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = aggr.SumAggregation()(x)
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
