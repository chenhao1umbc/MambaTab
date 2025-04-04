# %%
from main_test_lib import (
    read_data,
    create_dataloaders,
    test_result,
    train_test_split,
    train_model,
    train_ssl,
    MambaTab,
)
import torch

config = {
    "DATASET_NAME": "credit_approval",  # Please follow paper and use 'X' as dataset name, where X can be ='credit','dress', etc. As an example here X='credit_approval' is provided.
    "SEED": 15,  # variations in machine configurations can affect distributions
    "BATCH": 100,
    "LR": 0.0001,
    "EPOCH": 1000,
    "REPRESENTATION_LAYER": 32,
    "ssl_epochs": 100,
    "ssl_corruption": 0.5,
    "ssl": False,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
# Dataloading and split
x_data, y_data = read_data(dataset_name=config["DATASET_NAME"])
x_train, x_test, y_train, y_test = train_test_split(
    x_data,
    y_data,
    test_size=0.2,
    random_state=config["SEED"],
    stratify=y_data,
    shuffle=True,
)
# %%
val_size = int(len(y_data) * 0.1)
x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=val_size,
    random_state=config["SEED"],
    stratify=y_train,
    shuffle=True,
)

print("Train:", x_train.shape)
print("Val:", x_val.shape)
print("Test:", x_test.shape)

# Create dataloaders using the helper function
dataloader = create_dataloaders(
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    x_test=x_test,
    y_test=y_test,
    batch_size=config["BATCH"],
)

# Get the model
model = MambaTab(
    input_features=x_train.shape[1],
    n_class=1,
    intermediate_representation=config["REPRESENTATION_LAYER"],
)  # n_class=1 is to use a single output logit strategy, where n_class does not refer to the number of classes and is sufficient for binary classification
model = model.to(config["device"])

# SSL pretraining
if config["ssl"] == True:
    print("SSL pretraining")
    config["project_dim"] = x_train.shape[1]
    model = train_ssl(model=model, config=config, dataloader=dataloader)

# Train-validate the model
model = train_model(model, config, dataloader)

# Get test set performance
test_result(model, dataloader, config)
print("----------------Complete----------------")

# %%
