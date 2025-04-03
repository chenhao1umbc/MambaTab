# %%
import time
import torch
from main_test_lib import Mamba_pt

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

d_model = 128  # Example dimension
device = "mps"
model_pt = Mamba_pt(d_model=d_model).to(device)  # Instantiate your PyTorch Mamba
dummy_input = torch.randn(64, 200, d_model).to(device)  # (B, L, D)


# %%
print(f"Running on device: {device}")
print("Running without torch.compile...")
torch.mps.synchronize()
start_time = time.time()
output = model_pt(dummy_input)
torch.mps.synchronize()
second_run_time = time.time() - start_time
print(f"Used Time: {second_run_time:.4f} seconds")
print("Output shape:", output.shape)


# %%
