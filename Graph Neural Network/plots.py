import math

import numpy as np

from models import ModelArchitecture
import json
import matplotlib.pyplot as plt
from main import load_model, test, load_instances, print_dataset_distribution, read_instances
import torch


def get_loss_on(model, criterion,
                instance_folder: str, instance_names: list[str], extension: str):
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

        # file_name = instance.file_name

    #     print(f"{file_name} -> Target:{target.item()} , Prediction:{prediction.item():.4f} , Error:{errors[index]:.4f}")
    #
    # print(f"Total loss for files: {total_loss:.4f}")
    # print(f"Total MAE score: {np.sum(errors)}")
    # print(f"All errors: {errors}")

    return np.sum(errors)


# 3
def get_dataset_histograms():
    dataset_names = ["D1 10k N 30-60 E 7,5-20", "D2 10k N 30-60 3-6C", "D3 10k N 30-60 Clique", "D4 10k N30-60",
                     "D5 Hard"]
    for dataset_name in dataset_names:
        dataset = load_instances("Datasets", dataset_name)
        print_dataset_distribution(dataset, dataset_name)
        class_count = dict()
        for instance in dataset:
            value = class_count.setdefault(instance.chromatic_number, 0)
            class_count[instance.chromatic_number] = value + 1

        plt.hist(class_count)
        plt.xlabel("Chromatic Number")
        plt.ylabel("Count")
        dataset_s = dataset_name.split(" ")[0]
        plt.title(f"{dataset_s} Instances Distribution")

        plt.savefig(f"Plots/{dataset_s} Instances Distribution.png")
        plt.show()


# 2
def get_plot_best_models():
    test_instances_names = []
    test_instances_names += ["anna", "david", "huck", "jean", "homer"]
    # instances_names += ["zeroin.i.1", "zeroin.i.2", "zeroin.i.3"]
    # instances_names += ["games120", "miles250"]
    test_instances_names += ["queen5_5", "queen6_6", "queen7_7", "queen8_12", "queen8_8", "queen9_9", "queen13_13"]
    test_instances_names += ["myciel5", "myciel6", "myciel7"]
    test_instances_names += ["games120"]

    criterion = torch.nn.L1Loss()
    dataset_names = ["D1", "D2", "D3", "D4", "D5"]
    model_architectures = [ModelArchitecture.GraphConv, ModelArchitecture.SAGEConv, ModelArchitecture.SAGEConv_LSTM,
                           ModelArchitecture.GraphConv_LSTM]
    global_layer_aggregation = 'mean'
    plt.ylabel("Val Loss")
    plt.xlabel("No Epochs")
    x_values = np.arange(1, 50 + 1)
    for dataset_name in dataset_names:
        best_loss = math.inf
        best_val_loss = None
        best_name = None

        for model_architecture in model_architectures:
            model_architecture_str = str(model_architecture).split(".")[1]
            model_path = "Models/" + model_architecture_str + "-" + dataset_name + "-F1-" + global_layer_aggregation
            model_path_info = "Models/" + model_architecture_str + "-" + dataset_name + "-F1-" + global_layer_aggregation + "-Info.txt"
            model = load_model(model_path)

            test_loss = get_loss_on(model, criterion, "Instances", test_instances_names, ".col")
            if best_loss > test_loss:
                best_loss = test_loss
                with open(model_path_info) as file:
                    model_info = json.loads(file.read())
                best_val_loss = model_info['val_loss']
                best_name = model_architecture_str

        plt.plot(x_values, best_val_loss, label=dataset_name + "-" + best_name)

    plt.legend()
    plt.savefig("Plots/BestModelsValLoss.png")
    plt.show()


def create_all_datasets_plots():
    model_names = []
    model_architectures = [ModelArchitecture.GraphConv, ModelArchitecture.SAGEConv, ModelArchitecture.SAGEConv_LSTM,
                           ModelArchitecture.GraphConv_LSTM]
    dataset_names = ["D1", "D2", "D3", "D4", "D5"]
    global_layer_aggregation = 'mean'
    no_epochs = 50
    x_values = np.arange(1, no_epochs + 1)

    for dataset_name in dataset_names:
        plt.xlabel("No Epochs")
        plt.ylabel("Train Loss")
        plt.title(f"Train Loss over Epochs for {dataset_name}(with {global_layer_aggregation})")
        for model_architecture in model_architectures:
            model_architecture_str = str(model_architecture).split(".")[1]
            model_info_path = "Models/" + model_architecture_str + "-" + dataset_name + "-F1-" + global_layer_aggregation + "-Info.txt"
            with open(model_info_path) as file:
                model_info = json.loads(file.read())
            plt.plot(x_values, model_info['train_loss'], label=model_architecture_str)
        plt.legend()

        plt.savefig(f"Plots/{dataset_name}-Train Loss.png")

        plt.show()

        if dataset_name != "D5":
            plt.xlabel("No Epochs")
            plt.ylabel("Val Loss")
            plt.title(f"Val Loss over Epochs for {dataset_name}(with {global_layer_aggregation})")
            for model_architecture in model_architectures:
                model_architecture_str = str(model_architecture).split(".")[1]
                model_info_path = "Models/" + model_architecture_str + "-" + dataset_name + "-F1-" + global_layer_aggregation + "-Info.txt"
                with open(model_info_path) as file:
                    model_info = json.loads(file.read())
                plt.plot(x_values, model_info['val_loss'], label=model_architecture_str)
            plt.legend()

            plt.savefig(f"Plots/{dataset_name}-Val Loss.png")

            plt.show()


# 4
def create_table_code():
    pass


if __name__ == '__main__':
    create_all_datasets_plots()
