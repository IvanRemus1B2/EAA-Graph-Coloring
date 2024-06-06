import math

import numpy as np

from models import ModelArchitecture
import json
import matplotlib.pyplot as plt
from main import load_model, test, load_instances, print_dataset_distribution, read_instances
import torch


def read_val_loss(file_name: str):
    val_loss = []
    with open(file_name + ".txt") as file:
        for line in file.readlines():
            if "Epoch " in line:
                val_loss.append(float(line.split(",")[3].split("=")[1]))

    return np.array(val_loss)


def get_loss_on(model, criterion,
                instance_folder: str, instance_names: list[str], extension: str):
    model.eval()

    instances = read_instances(instance_names, instance_folder, extension)

    init_no_node_features = model.no_units_per_gc_layer[0]

    errors = np.zeros(len(instance_names))
    all_predictions = dict()
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

        all_predictions[instance.file_name] = prediction.item()

        # file_name = instance.file_name

    #     print(f"{file_name} -> Target:{target.item()} , Prediction:{prediction.item():.4f} , Error:{errors[index]:.4f}")
    #
    # print(f"Total loss for files: {total_loss:.4f}")
    # print(f"Total MAE score: {np.sum(errors)}")
    # print(f"All errors: {errors}")

    return np.sum(errors), all_predictions


# 3
def get_dataset_histograms():
    dataset_names = ["D1 10k N 30-60 E 7,5-20", "D2 10k N 30-60 3-6C", "D3 10k N 30-60 Clique", "D4 10k N30-60",
                     "D5 Hard"]
    for dataset_name in dataset_names:
        dataset = load_instances("Datasets", dataset_name)
        print_dataset_distribution(dataset, dataset_name)
        # values = []
        class_count = dict()
        for instance in dataset:
            value = class_count.setdefault(instance.chromatic_number, 0)
            class_count[instance.chromatic_number] = value + 1
            # values.append(instance.chromatic_number)

        values = []
        counts = []
        for value, count in class_count.items():
            values.append(value)
            counts.append(count)

        plt.bar(values, counts)
        # plt.hist(class_count, bins=list(class_count.keys()).sort(), edgecolor="black")
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
            model_path_info = "Models/" + model_architecture_str + "-" + dataset_name + "-F1-" + global_layer_aggregation
            model = load_model(model_path)

            test_loss, _ = get_loss_on(model, criterion, "Instances", test_instances_names, ".col")
            if best_loss > test_loss:
                best_loss = test_loss
                best_val_loss = read_val_loss(model_path_info)
                best_name = model_architecture_str

        plt.plot(x_values, best_val_loss, label=dataset_name + "-" + best_name)

    plt.legend()
    plt.savefig("Plots/BestModelsValLoss.png")
    plt.show()


# 1
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
                model_info_path = "Models/" + model_architecture_str + "-" + dataset_name + "-F1-" + global_layer_aggregation

                plt.plot(x_values, read_val_loss(model_info_path), label=model_architecture_str)
            plt.legend()

            plt.savefig(f"Plots/{dataset_name}-Val Loss.png")

            plt.show()


# 4
def create_table_code(dataset_names: list[str]):
    test_instances_names = []
    test_instances_names += ["anna", "david", "huck", "jean", "homer"]
    # instances_names += ["zeroin.i.1", "zeroin.i.2", "zeroin.i.3"]
    # instances_names += ["games120", "miles250"]
    test_instances_names += ["queen5_5", "queen6_6", "queen7_7", "queen8_12", "queen8_8", "queen9_9", "queen13_13"]
    test_instances_names += ["myciel5", "myciel6", "myciel7"]
    test_instances_names += ["games120"]

    instances = read_instances(test_instances_names, "Instances", ".col")
    instance_sizes = []
    for instance in instances:
        instance_sizes.append((instance.file_name, instance.graph.number_of_nodes(), instance.chromatic_number))

    model_architectures = [ModelArchitecture.GraphConv, ModelArchitecture.SAGEConv, ModelArchitecture.SAGEConv_LSTM,
                           ModelArchitecture.GraphConv_LSTM]
    # dataset_names = ["D1", "D2", "D3", "D4"]
    global_layer_aggregation = 'mean'
    criterion = torch.nn.L1Loss()
    all_model_test_predictions = dict()
    all_model_test_loss = dict()
    for model_architecture in model_architectures:
        best_loss = math.inf
        best_prediction_values = None
        best_from = None
        model_architecture_str = str(model_architecture).split(".")[1]
        for dataset_name in dataset_names:
            model_path = "Models/" + model_architecture_str + "-" + dataset_name + "-F1-" + global_layer_aggregation

            model = load_model(model_path)
            test_loss, all_test_predictions = get_loss_on(model, criterion, "Instances", test_instances_names, ".col")

            if best_loss > test_loss:
                best_loss = test_loss
                best_prediction_values = all_test_predictions
                best_from = dataset_name

        all_model_test_predictions[model_architecture_str + f"({best_from})"] = best_prediction_values
        all_model_test_loss[model_architecture_str + f"({best_from})"] = best_loss

    models = list(all_model_test_predictions.keys())
    # print(models)

    print("\\begin{table}[H]")
    print("\\centering")
    print("\\resizebox{\\textwidth}{!}{")
    print("\\begin{tabular}{|c|c|c|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{Instances} & Size & \\chi & \\multicolumn{4}{c|}{Computed \\chi} \\\\")
    name3 = models[3].replace("_", "\\_")
    name2 = models[2].replace("_", "\\_")
    print(f"& & & {models[0]} & {models[1]} & {name3} & {name2} \\\\")
    print("\\hline")

    names = [models[0], models[1], models[3], models[2]]
    instance_sizes.sort(key=lambda instance: instance[1])
    for file_name, no_nodes, chromatic_number in instance_sizes:
        to_show = file_name
        if '_' in file_name:
            to_show = to_show.replace("_", "\\_")
        print(
            f"{to_show} & {no_nodes} & {chromatic_number}", end="")
        for name in names:
            pred = int(all_model_test_predictions[name][file_name] + 0.5)
            print(" & ", end="")
            if chromatic_number == pred:
                print("\\textbf{", pred, "}", end="")
            else:
                print(pred, end="")
        print("\\\\")
        print("\\hline")
    print("\\hline")
    print(
        f"MAE Loss & - & - & {all_model_test_loss[names[0]]:.4f} & {all_model_test_loss[names[1]]:.4f} & {all_model_test_loss[names[2]]:.4f} & {all_model_test_loss[names[3]]:.4f} \\\\")
    print("\\hline")
    print("\\end{tabular}}")
    print(
        "\\caption{Chromatic number by GNN obtained for the best models,where each model was selected as the best from all the models trained on a dataset}")
    print("\\end{table}")

    # print(instance_sizes)
    # print(all_model_test_predictions)


if __name__ == '__main__':
    # print(read_val_loss("SAGEConv_LSTM-D5-F1-mean"))
    # create_all_datasets_plots()
    # create_table_code(["D1", "D2", "D3", "D4"])
    # create_table_code(["D5"])

    # get_dataset_histograms()

    # create_all_datasets_plots()
    get_plot_best_models()
