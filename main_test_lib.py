# Standard libraries
import numpy as np
import pandas as pd
import copy
import tqdm
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from pandas.api.types import is_string_dtype

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba_pt(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        # Input projection
        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )

        # 1D Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv
            - 1,  # Causal padding: output[:, :, i] only depends on input[:, :, :i+1]
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        # Projections for SSM parameters (delta, B, C)
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        # Projection for delta (dt)
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        # Initialize dt projection
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverse of softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization for A matrix
        A = torch.arange(1, self.d_state + 1).to(torch.float).repeat(self.d_inner, 1)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D) -> (Batch, Length, Dim)
        Returns: same shape as hidden_states
        """
        B, L, D = hidden_states.shape

        conv_state, ssm_state = None, None

        # 1. Input Projection
        # (B, L, D) -> (B, L, 2 * D_in)
        xz = self.in_proj(hidden_states)
        # (B, L, 2 * D_in) -> 2 * (B, L, D_in)
        x, z = xz.chunk(2, dim=-1)

        # 2. 1D Convolution
        # (B, L, D_in) -> (B, D_in, L)
        x = x.permute(0, 2, 1)

        # Apply causal convolution
        # conv1d requires (B, C, L) input. padding=d_conv-1 makes it causal
        # Output shape (B, D_in, L)
        x_conv = self.conv1d(x)

        # Remove the future padding added by conv1d
        # (B, D_in, L)
        x_conv = x_conv[:, :, :L]

        # Apply activation
        # (B, D_in, L)
        x_activated = self.act(x_conv)

        # 3. SSM Calculation (Implemented via loop - less efficient)
        # (B, D_in, L) -> (B, L, D_in)
        x_activated = x_activated.permute(0, 2, 1)

        # Project for SSM parameters
        # (B, L, D_in) -> (B * L, D_in)
        x_flat = x_activated.reshape(B * L, self.d_inner)
        # (B * L, D_in) -> (B * L, dt_rank + 2 * d_state)
        x_proj = self.x_proj(x_flat)

        # Split into dt, B, C
        # dt: (B * L, dt_rank), B: (B * L, d_state), C: (B * L, d_state)
        dt_pre, B_pre, C_pre = torch.split(
            x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Calculate dt
        # (B * L, dt_rank) -> (dt_rank, B * L)
        dt_t = dt_pre.t()
        # (d_inner, dt_rank) @ (dt_rank, B * L) -> (d_inner, B * L)
        dt_unbiased = self.dt_proj.weight @ dt_t
        # (d_inner, B*L) -> (B*L, d_inner) -> (B, L, d_inner)
        dt_biased = dt_unbiased.t().reshape(
            B, L, self.d_inner
        ) + self.dt_proj.bias.view(1, 1, -1)
        # Apply softplus to ensure positivity
        # dt: (B, L, d_inner)
        dt = F.softplus(dt_biased)

        # Reshape B and C
        # B: (B*L, d_state) -> (B, L, d_state)
        B_ssm = B_pre.reshape(B, L, self.d_state)
        # C: (B*L, d_state) -> (B, L, d_state)
        C_ssm = C_pre.reshape(B, L, self.d_state)

        # Get A matrix (fixed, not input-dependent)
        # (d_inner, d_state)
        A = -torch.exp(self.A_log.float())

        # Initialize SSM state (hidden state h)
        # (B, D_in, d_state)
        ssm_state = torch.zeros(
            B, self.d_inner, self.d_state, device=hidden_states.device
        )
        ys = []

        # Iterate over sequence length L (Recurrence)
        # This loop is the core difference and the source of inefficiency
        for i in range(L):
            # Get parameters for this timestep
            dt_i = dt[:, i, :]  # (B, d_inner)
            B_i = B_ssm[:, i, :]  # (B, d_state)
            C_i = C_ssm[:, i, :]  # (B, d_state)
            x_i = x_activated[:, i, :]  # (B, d_inner)

            # Discretize A and B
            # A_bar = exp(dt * A)
            # dA_i: (B, d_inner, d_state)
            dA_i = torch.exp(torch.einsum("bi,in->bin", dt_i, A))
            # dB_i = dt * B
            # dB_i: (B, d_inner, d_state) - Requires B to be shaped correctly for broadcast/einsum
            # We need elementwise product of dt_i (B, d_inner) and B_i (B, d_state) broadcasted? No.
            # dB = dt * B (according to paper's discretization)
            # Need to multiply dt_i (B, d_inner) with B_i (B, d_state) elementwise after expansion?
            # Let's follow the logic from original step fn more closely: dB = einsum("bd,bn->bdn", dt, B)
            # But B here is input-dependent B_ssm.
            # dB_i = torch.einsum('bi,bn->bin', dt_i, B_i) # This seems wrong dim for B_i

            # Looking at the original Mamba forward pass structure (non-fused):
            # B comes from x_proj, reshaped to (B, L, d_state)
            # dt comes from dt_proj, reshaped to (B, L, d_inner)
            # Need dB = (dt * B) where B relates to ssm input x.
            # The equation h_t = A_bar * h_{t-1} + B_bar * x_t requires B_bar = (dA - I) * A^{-1} * B approx dt * B
            # Let's use the simplified discretization B_bar approx dt * B
            # dB_i needs shape (B, d_inner, d_state)
            # x_i needs shape (B, d_inner)
            # Original step: dB = torch.einsum("bd,bn->bdn", dt, B) - Here B is d_state vector per d_inner
            dB_i = torch.einsum(
                "bi,bis->bis", dt_i, B_i.unsqueeze(1).expand(-1, self.d_inner, -1)
            )  # Tentative based on step logic

            # Update SSM state: h_t = dA * h_{t-1} + dB * x_t
            # x_t shape is (B, d_inner), need to unsqueeze for matmul with dB?
            # Original step: ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            # So, x needs to be (B, d_inner, 1)
            ssm_state = ssm_state * dA_i + dB_i * x_i.unsqueeze(
                -1
            )  # (B, d_inner, d_state)

            # Calculate output: y_t = C * h_t
            # C_i is (B, d_state). Need C per d_inner? Yes, C_ssm is (B, L, d_state)
            # Original step: y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C) - C is d_state per d_inner
            C_i_eff = C_ssm[:, i, :]  # (B, d_state)
            # y_i needs shape (B, d_inner)
            # Need C_i (B, d_inner, d_state) ?
            # Let's assume C_ssm was meant to be (B, L, d_inner, d_state) ? No, (B,L,d_state) seems right from split.
            # If y = C*h, where h is (B, D_in, d_state), then C must be (B, D_in, d_state) -> (B, D_in) ?
            # Original selective_scan C is (B, d_state, L) -> C_i is (B, d_state)
            # Original step C is (B, d_state)
            # torch.einsum("bdn,bn->bd", ssm_state, C_i) fails dimension check
            # Let's re-read S4/S6/Mamba... Output y = C h(t)
            # If h is (B, d_inner, d_state), C should be (B, d_inner, d_state) to map state back to d_inner.
            # Assume C_ssm from x_proj gives (B, L, d_state) and this needs transforming?
            # Or maybe the einsum in the original step is ('bdn,bdn->bd') if C is structured differently?
            # Let's assume C_pre (B*L, d_state) should be reshaped to (B, L, d_state) and somehow used.

            # Revisit Mamba paper/code: C is input-dependent, typically projected from x like B.
            # The selective_scan_fn expects C of shape (B, d_state, L) or similar.
            # The implementation here got C_ssm as (B, L, d_state). Let's assume this is correct.
            # How to combine (B, d_inner, d_state) state with (B, d_state) C to get (B, d_inner) y?
            # Perhaps y_i = torch.einsum('bis,bs->bi', ssm_state, C_i) ? This works dimensionally.
            y_i = torch.einsum(
                "bin,bn->bi", ssm_state, C_i
            )  # Matches structure if C is shared across d_inner

            ys.append(y_i)

        # Stack outputs from loop
        # (L, B, D_in) -> (B, L, D_in)
        y = torch.stack(ys, dim=1)

        # Add D skip connection y = y + D * x
        # x_activated is (B, L, D_in)
        y = y + x_activated * self.D.view(1, 1, -1)

        # Multiply by gating function z
        # y: (B, L, D_in), z: (B, L, D_in)
        output = y * self.act(z)

        # 4. Output Projection
        # (B, L, D_in) -> (B, L, D)
        output = self.out_proj(output)

        return output


class MambaTab(torch.nn.Module):
    """
    This class defines the MambaTab model
    """

    def __init__(
        self,
        input_features,
        n_class,
        intermediate_representation,  # config["REPRESENTATION_LAYER"],
    ):
        super(MambaTab, self).__init__()
        self.linear_layer = torch.nn.Linear(input_features, intermediate_representation)
        self.relu = torch.nn.ReLU()
        self.layer_norm = torch.nn.LayerNorm(intermediate_representation)

        self.mamba = Mamba(
            d_model=intermediate_representation, d_state=32, d_conv=4, expand=2
        )  # Please use different parameters settings for different configurations
        self.output_layer = torch.nn.Linear(intermediate_representation, n_class)

    def forward(self, x):
        x = self.linear_layer(x)
        x = self.layer_norm(x)
        x = self.relu(x)
        x = self.mamba(x)
        x = self.output_layer(x)
        return x


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


def test_result(model, dataloader, config):
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


def train_model(model, config, dataloader):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    early_stopping_counter = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=config["LR"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["EPOCH"], eta_min=0, verbose=False
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in tqdm.tqdm(range(config["EPOCH"])):
        if early_stopping_counter >= 5:
            break
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            metrics = defaultdict(float)
            epoch_samples = 0

            for btch, feed_dict in enumerate(dataloader[phase]):
                inputs = feed_dict[0]
                inputs = inputs.unsqueeze(0)
                labels = feed_dict[1]

                inputs = inputs.type(torch.FloatTensor)
                inputs = inputs.to(config["device"])
                labels = labels.type(torch.FloatTensor)
                labels = labels.to(config["device"])

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    outputs = outputs.squeeze()
                    loss = loss_fn(outputs, labels)
                    metrics["loss"] += loss.item()
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                epoch_samples += 1
            epoch_loss = metrics["loss"] / epoch_samples

            if phase == "val":
                if epoch_loss < best_loss:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_loss = epoch_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

        scheduler.step()
    model.load_state_dict(best_model_wts)
    return model


def train_ssl(model, config, dataloader):
    train_losses = []
    val_losses = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    # Change model's last layer:
    model.output_layer = torch.nn.Linear(
        config["REPRESENTATION_LAYER"], config["project_dim"]
    )
    model = model.to(config["device"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["LR"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["EPOCH"], eta_min=0, verbose=False
    )
    ssl_loss = torch.nn.MSELoss()

    for epoch in tqdm.tqdm(range(config["ssl_epochs"])):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            metrics = defaultdict(float)
            epoch_samples = 0

            for btch, feed_dict in enumerate(dataloader[phase]):
                inputs = feed_dict[0]

                inputs = inputs.type(torch.FloatTensor)
                num_elements = int(torch.prod(torch.tensor(inputs.shape)))
                # Determine the number of zeros and ones
                num_zeros = int(num_elements * config["ssl_corruption"])
                num_ones = num_elements - num_zeros
                tensor_zeros = torch.zeros(num_zeros)  # Create a tensor of zeros
                tensor_ones = torch.ones(num_ones)  # Create a tensor of ones

                # Concatenate the tensors of zeros and ones
                tensor_data = torch.cat((tensor_zeros, tensor_ones))

                # Shuffle the tensor
                tensor_shuffled = tensor_data[torch.randperm(num_elements)].reshape(
                    inputs.shape
                )
                to_predict = inputs.detach().clone()
                inputs = tensor_shuffled * inputs
                inputs = inputs.unsqueeze(0)

                inputs = inputs.to(config["device"])
                to_predict = to_predict.to(config["device"])
                to_predict = to_predict.unsqueeze(0)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):

                    predicted = model(inputs)

                    loss = ssl_loss(predicted, to_predict)
                    metrics["loss"] += loss.item() * inputs.size(0)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)
            epoch_loss = metrics["loss"] / epoch_samples

            if phase == "val":
                if epoch_loss < best_loss:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_loss = epoch_loss

                val_losses.append(epoch_loss)
            else:
                train_losses.append(epoch_loss)

        scheduler.step()
    model.load_state_dict(best_model_wts)
    # Change back to classification layer
    model.output_layer = torch.nn.Linear(config["REPRESENTATION_LAYER"], 1)
    model = model.to(config["device"])

    return model
