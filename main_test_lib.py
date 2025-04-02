import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from pandas.api.types import is_string_dtype
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
import torch
from config import config
from sklearn.model_selection import train_test_split


class TabularDataLoader(Dataset):
    """
    This is pytorch dataloader. It gives input to the model for train/val/test
    """

    def __init__(
        self,
        length,
        data_type,
        x_train=None,
        y_train=None,
        x_val=None,
        y_val=None,
        x_test=None,
        y_test=None,
    ):
        self.length = length
        self.data_type = data_type
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.data_type == "train":
            return self.x_train[idx], self.y_train[idx]
        if self.data_type == "val":
            return self.x_val[idx], self.y_val[idx]
        if self.data_type == "test":
            return self.x_test[idx], self.y_test[idx]


def read_data(dataset_name):
    data = pd.read_csv("datasets/" + dataset_name + "/data_processed" + ".csv")

    # fill nulll values
    for col in data.columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # categorical encoder
    for c in data.columns:
        if is_string_dtype(data[c]):
            data[c] = data[c].str.lower()
            enc = OrdinalEncoder()
            cur_data = np.array(data[c])
            cur_data = np.reshape(cur_data, (cur_data.shape[0], 1))
            data[c] = enc.fit_transform(cur_data)

    y_data = data[data.columns[-1]]
    x_data = data.drop(labels=[data.columns[-1]], axis=1)
    x_data = MinMaxScaler().fit_transform(x_data)
    x_data, y_data = np.array(x_data), np.array(y_data)
    return x_data, y_data


def test_result(model, dataloader):
    """
    This function is for inference on the test set.
    """
    model.eval()
    all_test_output_probas = []
    all_test_labels = []
    sig = torch.nn.Sigmoid()

    for inputs, labels in dataloader["test"]:
        inputs = inputs.unsqueeze(0)
        inputs = inputs.type(torch.FloatTensor)
        inputs = inputs.to(config["device"])

        labels = labels.to(config["device"])
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs = sig(outputs)
            outputs = outputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            for i in range(outputs.shape[0]):
                all_test_labels.append(labels[i])
                all_test_output_probas.append(outputs[i])
    performance_value = roc_auc_score(all_test_labels, all_test_output_probas)
    print("AUROC score: ", performance_value)
    return performance_value


def create_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size):
    """
    Create and return dataloaders for training, validation and testing
    """
    train_set = TabularDataLoader(
        length=x_train.shape[0],
        data_type="train",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )

    val_set = TabularDataLoader(
        length=x_val.shape[0],
        data_type="val",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )

    test_set = TabularDataLoader(
        length=x_test.shape[0],
        data_type="test",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )

    dataloader = {
        "train": torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=0
        ),
        "val": torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, shuffle=False, num_workers=0
        ),
        "test": torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=0
        ),
    }

    return dataloader


def run_feature_incremental(initial_model, train_model_func, dataset_name, config):
    """
    Run the feature incremental learning process.
    Incrementally increases the number of features used for training.

    Args:
        initial_model: Initial model architecture
        train_model_func: Function to train the model
        dataset_name: Name of the dataset to use
        config: Configuration dictionary

    Returns:
        The trained model
    """
    model = initial_model

    for incremental in range(3):
        # Load and split data
        x_data, y_data = read_data(dataset_name=dataset_name)
        x_train, x_test, y_train, y_test = train_test_split(
            x_data,
            y_data,
            test_size=0.2,
            random_state=config["SEED"],
            stratify=y_data,
            shuffle=True,
        )  # Fixed-seed split. so no overlap in iterations
        val_size = int(len(y_data) * 0.1)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=val_size,
            random_state=config["SEED"],
            stratify=y_train,
            shuffle=True,
        )

        # If not the last iteration, use a subset of features
        if incremental != 2:
            subset_size = x_train.shape[1] // 3
            x_train = x_train[:, 0 : subset_size * (incremental + 1)]
            x_val = x_val[:, 0 : subset_size * (incremental + 1)]
            x_test = x_test[
                :, 0 : subset_size * (incremental + 1)
            ]  # Also subset test data

        print("Subset size:", x_train.shape[1])

        # Create dataloaders
        dataloader = create_dataloaders(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            batch_size=config["BATCH"],
        )

        # Update model's input layer to match the current feature size
        model.linear_layer = torch.nn.Linear(
            x_train.shape[1], config["REPRESENTATION_LAYER"]
        )  # Adapt the first layer's input shape for feature incremental learning

        model = model.to(config["device"])

        # Train the model
        model = train_model_func(model, config, dataloader)

        # Evaluate on test set
        performance = test_result(model, dataloader)
        print(
            f"Incremental step {incremental + 1}/3 complete. Performance: {performance:.4f}"
        )

    return model
