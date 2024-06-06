from models import ModelArchitecture
import json


def create_all_datasets_plots():
    model_names = []
    model_architectures = [ModelArchitecture.GraphConv, ModelArchitecture.SAGEConv, ModelArchitecture.SAGEConv_LSTM,
                           ModelArchitecture.GraphConv_LSTM]
    dataset_names = ["D1", "D2", "D3", "D4", "D5"]
    global_layer_aggregation = 'mean'
    no_epochs = 50

    for dataset_name in dataset_names:
        model_info = []
        for model_architecture in model_architectures:
            model_info_path = "Models/" + str(model_architecture).split(".")[
                1] + "-" + dataset_name + "-F1-" + global_layer_aggregation + "-Info.txt"
            with open(model_info_path) as file:
                model_info.append(json.loads(file.read()))

        if dataset_name != "D5":
            pass


if __name__ == '__main__':
    create_all_datasets_plots()
