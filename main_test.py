from config import config
from sklearn.model_selection import train_test_split
from train_val import train_model, train_ssl
from MambaTab import MambaTab
from main_test_lib import read_data, create_dataloaders, test_result

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
    input_features=x_train.shape[1], n_class=1
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
test_result(model, dataloader)
print("----------------Complete----------------")
